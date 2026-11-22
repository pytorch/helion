"""GPU tests of ``cute_guarded_lanes`` and ``cute_fmnmx_extrema``: lanes run
under their stores' guard, predicated packets and single-instruction fp32
extrema store what the knob-off kernels and eager PyTorch do, on partial
tiles, row lengths that take no packet, packets of 8, 16 and 32 bytes of
every element type that loads in packets, padded windows, views that are
offset, strided or permuted, an output that is its own masked input, NaNs
and signed zeros."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.guarded_lane_kernels import IDENTITY_CHAINS
from helion._testing.guarded_lane_kernels import LITERAL_IDENTITY_CHAINS
from helion._testing.guarded_lane_kernels import extrema
from helion._testing.guarded_lane_kernels import extrema_bound_chains
from helion._testing.guarded_lane_kernels import extrema_carried_max
from helion._testing.guarded_lane_kernels import extrema_carried_max_copies
from helion._testing.guarded_lane_kernels import extrema_carried_max_identities
from helion._testing.guarded_lane_kernels import extrema_chains
from helion._testing.guarded_lane_kernels import extrema_identity_chains
from helion._testing.guarded_lane_kernels import extrema_literal_identity_chains
from helion._testing.guarded_lane_kernels import extrema_outside_chains
from helion._testing.guarded_lane_kernels import extrema_relu_clamp
from helion._testing.guarded_lane_kernels import extrema_view_chains
from helion._testing.guarded_lane_kernels import flags_reread_after_a_store
from helion._testing.guarded_lane_kernels import masked_accumulate
from helion._testing.guarded_lane_kernels import neighbor_max_into
from helion._testing.guarded_lane_kernels import packet_tile_seeds
from helion._testing.guarded_lane_kernels import padded_and_transposed
from helion._testing.guarded_lane_kernels import padded_sum
from helion._testing.guarded_lane_kernels import padded_sum_reference
from helion._testing.guarded_lane_kernels import pooled_max
from helion._testing.guarded_lane_kernels import pooled_max_reference
from helion._testing.guarded_lane_kernels import window_max
from helion._testing.guarded_lane_kernels import window_max_into
from helion._testing.guarded_lane_kernels import window_max_reference

# 16-byte packets along the channels, four threads on each of H and W: the
# last tiles of H and W are partial at every shape below.
_PACKETS = {
    "block_sizes": [4, 4, 16],
    "num_threads": [4, 4, 2],
    "cute_vector_widths": [1, 1, 8],
}
_KNOBS = {
    "off": {},
    "guarded": {"cute_guarded_lanes": True},
    "guarded_fmnmx": {"cute_guarded_lanes": True, "cute_fmnmx_extrema": True},
}
_KERNELS: dict[tuple[object, str, str], object] = {}


def _kernel(kernel: object, config: dict[str, object], knobs: str) -> object:
    """``kernel`` at ``config`` with the knobs of ``_KNOBS[knobs]``, built
    once per module."""
    key = (kernel, repr(sorted(config.items())), knobs)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            kernel.fn,  # type: ignore[attr-defined]
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict({**config, **_KNOBS[knobs]}),
        )
    return _KERNELS[key]


def _seed_kernel(kernel: object, seed: dict[str, object]) -> object:
    """``kernel`` at a compiler seed as the seed is, its knobs its own,
    built once per module."""
    return _kernel(kernel, seed, "off")


def _search_seeds(
    kernel: object, args: tuple[torch.Tensor, ...]
) -> list[dict[str, object]]:
    """The packet tile seeds that the search of ``kernel`` at ``args`` starts
    from."""
    search = helion.kernel(
        kernel.fn,  # type: ignore[attr-defined]
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
    )
    return packet_tile_seeds(search.bind(args).config_spec)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(
        {2: torch.int16, 4: torch.int32}[value.element_size()]
    )


def _assert_same_values(
    actual: torch.Tensor, expected: torch.Tensor, what: str
) -> None:
    """NaNs in the same places and the same bits everywhere else.  (Which
    NaN a float conversion or an extremum yields is not part of the result:
    without the lane selects the assembler narrows an fp32 chain over
    widened half lanes to half-precision instructions, whose NaN differs in
    its sign bit.)"""
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan), what
    assert torch.equal(_bits(actual)[~nan], _bits(expected)[~nan]), what


def _activation(
    shape: tuple[int, ...],
    *,
    specials: bool,
    dtype: torch.dtype = torch.bfloat16,
    spread: float = 1.0,
) -> torch.Tensor:
    """An activation without zeros (so the sign of a zero maximum cannot
    differ), with NaNs and infinities among its values on request (a float
    ``dtype`` only)."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=generator) * spread + 3.0
    x = torch.where(x.abs() < 0.01, torch.ones_like(x), x)
    if specials:
        flat = x.view(-1)
        flat[::7] = float("nan")
        flat[3::11] = float("inf")
        flat[5::13] = float("-inf")
    return x.to(dtype).to(DEVICE)


def _views(name: str) -> torch.Tensor:
    """The activation as a view of a larger tensor: a contiguous window
    that starts one row into its storage, every other row of it, or the
    channels-last permutation of a channels-first tensor (whose channels are
    not contiguous, so no packet loads)."""
    base = _activation((38, 14, 16), specials=True)
    if name == "offset":
        return base[2:20]
    if name == "strided":
        return base[2::2]
    assert name == "permuted"
    return _activation((16, 18, 14), specials=True).permute(1, 2, 0)


@pytest.mark.parametrize(
    ("shape", "config"),
    [
        # Packets; 9 x 7 outputs in tiles of 4 x 4.
        ((18, 14, 16), _PACKETS),
        # 24 channels: three 16-byte packets, in tiles of one.
        (
            (18, 14, 24),
            {
                "block_sizes": [4, 4, 8],
                "num_threads": [4, 4, 1],
                "cute_vector_widths": [1, 1, 8],
            },
        ),
        # 20 channels take no 16-byte packet: scalar loads, partial tiles on
        # every axis.
        ((18, 14, 20), {"block_sizes": [4, 4, 8], "num_threads": [4, 4, 8]}),
    ],
    ids=["packets", "packet_tiles", "scalar"],
)
def test_window_maxima_match_eager(
    shape: tuple[int, int, int], config: dict[str, object]
) -> None:
    x = _activation(shape, specials=True)
    expected = window_max_reference(x)
    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    for knobs in _KNOBS:
        actual = _kernel(window_max, config, knobs)(x)  # type: ignore[operator]
        # Bit for bit where no NaN is: no window holds zeros of both signs.
        _assert_same_values(actual, expected, knobs)


# Every element type that loads in packets of 16- or 32-bit words (fp16,
# bf16 and int16 lanes are halves of a word; fp32 lanes are words) x the
# bytes of a lane's packet (a load of two, four or eight words).  The odd
# half of a predicated packet's word is where ptxas widens an fp16 lane from
# its neighbour (``lane_packets.PREDICATED_LANE_DTYPES``); int16 converts
# from the same halves and bf16 through other instructions, at each width.
@pytest.mark.parametrize("packet_bytes", [8, 16, 32])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.int16, torch.float32],
    ids=["fp16", "bf16", "int16", "fp32"],
)
def test_window_maxima_of_every_packet_match_eager(
    dtype: torch.dtype, packet_bytes: int
) -> None:
    """Partial tiles on all three axes (9 x 7 outputs in tiles of 4 x 4, 48
    channels in tiles of 32): the store is guarded by the mask of a
    packet's last lane and the packets load ahead of it, predicated where
    their dtype does."""
    width = packet_bytes // dtype.itemsize
    config = {
        "block_sizes": [4, 4, 32],
        "num_threads": [4, 4, 32 // width],
        "cute_vector_widths": [1, 1, width],
    }
    # Values a few units apart, so most windows' maxima differ from the
    # window's last element and from their neighbours'.
    x = _activation(
        (18, 14, 48),
        specials=dtype.is_floating_point,
        dtype=dtype,
        spread=1.0 if dtype.is_floating_point else 100.0,
    )
    expected = window_max_reference(x)
    # Both extremum spellings at one width; the compare-and-select one,
    # which is where fp16 went wrong, at every width.
    for knobs in ("guarded", "guarded_fmnmx") if packet_bytes == 16 else ("guarded",):
        actual = _kernel(window_max, config, knobs)(x)  # type: ignore[operator]
        _assert_same_values(actual, expected, knobs)


def test_fp16_window_maxima_under_the_store_guard_match_eager() -> None:
    """Whole 16-byte fp16 packets, partial tiles on H and W only: the pass
    moves the plain loads under the stores' guard and their lanes take no
    select, the form nearest to the predicated fp16 lanes that ptxas widens
    from the wrong half (``lane_packets.PREDICATED_LANE_DTYPES``)."""
    x = _activation((18, 14, 16), specials=True, dtype=torch.float16)
    expected = window_max_reference(x)
    for knobs in _KNOBS:
        actual = _kernel(window_max, _PACKETS, knobs)(x)  # type: ignore[operator]
        _assert_same_values(actual, expected, knobs)


def _pooled_input(case: str) -> torch.Tensor:
    """A ``[35, 39, C]`` activation with NaNs and infinities (17 x 19
    outputs of a 3x3 window: partial tiles of four on H and W): contiguous
    with 64 channels (eight 16-byte bf16 packets a row), in fp32, with 24
    channels (three packets, a tile of one), five elements into its storage
    (no packet starts on a 16-byte boundary), or every other row of a
    taller one."""
    if case == "fp32":
        return _activation((35, 39, 64), specials=True, dtype=torch.float32)
    if case == "three_packets":
        return _activation((35, 39, 24), specials=True)
    if case == "offset":
        return _activation((35 * 39 * 64 + 5,), specials=True)[5:].view(35, 39, 64)
    if case == "strided":
        return _activation((70, 39, 64), specials=True)[::2]
    assert case == "bf16"
    return _activation((35, 39, 64), specials=True)


@pytest.mark.parametrize("case", ["bf16", "fp32", "three_packets", "offset", "strided"])
def test_pooled_maxima_at_the_seeded_cta_minimum_match_eager(case: str) -> None:
    """The packet tile seed that names ``cute_min_blocks_per_mp``, taken
    from the kernel's own seeds: a 3x3 max pooling, whose lanes fold nine
    packets, stores eager's values at it, and with the minimum raised to 4."""
    x = _pooled_input(case)
    expected = pooled_max_reference(x, 3, 3)
    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    spec = (
        helion.kernel(
            pooled_max.fn, backend="cute", static_shapes=True, autotune_effort="full"
        )
        .bind((x, 3, 3))
        .config_spec
    )
    [twin] = [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_guarded_lanes") is True
        and "cute_min_blocks_per_mp" in seed.config
    ]
    assert twin["cute_min_blocks_per_mp"] == 1
    for minimum in (1, 4) if case == "bf16" else (1,):
        config = {**twin, "cute_min_blocks_per_mp": minimum}
        actual = _kernel(pooled_max, config, "off")(x, 3, 3)  # type: ignore[operator]
        _assert_same_values(actual, expected, f"{case}, minimum {minimum}")


@pytest.mark.parametrize("view", ["offset", "strided", "permuted"])
def test_window_maxima_of_views_match_eager(view: str) -> None:
    x = _views(view)
    assert not x.is_contiguous() or x.storage_offset() > 0
    actual = _kernel(window_max, _PACKETS, "guarded_fmnmx")(x)  # type: ignore[operator]
    _assert_same_values(actual, window_max_reference(x), view)


@pytest.mark.parametrize(
    ("dims", "seed"),
    [((1, 2, 0), 0), ((2, 1, 0), 2)],
    ids=["channel_first", "by_columns"],
)
def test_window_maxima_into_another_layout_match_eager_at_their_seeds(
    dims: tuple[int, int, int], seed: int
) -> None:
    """A window stored into a ``[9, 7, C]`` view of a buffer of another
    layout (``[C, H, W]``, or ``[C, W, H]``, its planes stored column by
    column), at the packet tile seeds the kernel's own search starts from:
    packets along the channels on thread x, the threads behind them along
    the buffer's stride-1 dim, a store per lane value.  The first shape into
    the channel-first buffer and the sibling (every other thread on the
    stride-1 dim) into the other; the tiles on H and W are partial."""
    x = _activation((18, 14, 64), specials=True)
    expected = window_max_reference(x)
    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    shape = [expected.shape[dims.index(dim)] for dim in range(3)]
    buffer = torch.zeros(shape, dtype=x.dtype, device=x.device)
    out = buffer.permute(dims)
    assert out.shape == expected.shape and out.stride(2) != 1
    seeds = _search_seeds(window_max_into, (x, out))
    assert len(seeds) == 3
    config = seeds[seed]
    assert config["cute_vector_widths"] == [1, 1, 8]
    # The packet block leads the loop order, then the buffer's stride-1 dim.
    assert config["loop_orders"][0][:2] == [2, dims.index(2)]
    _seed_kernel(window_max_into, config)(x, out)  # type: ignore[operator]
    _assert_same_values(out, expected, f"{dims}, seed {seed}")


def test_neighbor_maxima_into_a_channel_first_buffer_match_eager_at_their_seed() -> (
    None
):
    """Two taps a column apart, the second written as the tile's index plus
    one, stored into a ``[9, 7, C]`` view of a ``[C, H, W]`` buffer at the
    first packet tile seed of the kernel's own search.  The taps keep their
    tiles in the loads' facts, so the seed is there for the two loads per
    store, not for a computed coordinate.  Partial tiles on H and W."""
    x = _activation((9, 8, 64), specials=True)
    value = x.to(torch.float32)
    expected = torch.maximum(value[:, :-1], value[:, 1:]).to(x.dtype)
    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    buffer = torch.zeros([64, 9, 7], dtype=x.dtype, device=x.device)
    out = buffer.permute(1, 2, 0)
    seeds = _search_seeds(neighbor_max_into, (x, out))
    assert len(seeds) == 3
    config = seeds[0]
    assert config["cute_vector_widths"] == [1, 1, 8]
    assert config["loop_orders"] == [[2, 1, 0]]
    _seed_kernel(neighbor_max_into, config)(x, out)  # type: ignore[operator]
    _assert_same_values(out, expected, "neighbors")


@pytest.mark.parametrize(
    ("dtype", "channels", "width"),
    [
        (torch.bfloat16, 16, 8),
        (torch.bfloat16, 32, 16),
        # fp16 packets stay anchored, their lanes selecting under the guard.
        (torch.float16, 32, 16),
        (torch.float32, 16, 8),
    ],
    ids=["bf16_16_bytes", "bf16_32_bytes", "fp16_32_bytes", "fp32_32_bytes"],
)
def test_padded_sums_match_eager(dtype: torch.dtype, channels: int, width: int) -> None:
    config = {
        "block_sizes": [4, 4, channels],
        "num_threads": [4, 4, channels // width],
        "cute_vector_widths": [1, 1, width],
        # An L2 policy, an L1 priority and no hint on the three loads.
        "load_eviction_policies": ["l1_l2_last", "last", ""],
    }
    x = _activation((9, 7, channels), specials=True, dtype=dtype)
    expected = padded_sum_reference(x)
    results = {
        knobs: _kernel(padded_sum, config, knobs)(x)  # type: ignore[operator]
        for knobs in ("off", "guarded")
    }
    for knobs, actual in results.items():
        # Sums of three fp32 values in one order: no contraction, so the
        # stored bits are eager's.
        _assert_same_values(actual, expected, knobs)


def test_lane_values_reach_a_packet_store_and_a_per_lane_store() -> None:
    """A padded row written as it is (a packet per lane group) and
    transposed (a store per lane): 48 columns in tiles of 32, so the last
    tile's threads past the row skip both stores."""
    config = {
        "block_sizes": [1, 32],
        "num_threads": [0, 4],
        "cute_vector_widths": [1, 8],
    }
    generator = torch.Generator().manual_seed(2)
    x = torch.randn((9, 41), generator=generator).to(torch.bfloat16).to(DEVICE)
    expected = torch.nn.functional.pad(x, (0, 7))
    for knobs in ("off", "guarded"):
        out = torch.full_like(expected, 5.0)
        out_t = torch.full_like(expected.t(), 5.0).contiguous()
        _kernel(padded_and_transposed, config, knobs)(x, out, out_t)  # type: ignore[operator]
        assert torch.equal(_bits(out), _bits(expected)), knobs
        assert torch.equal(_bits(out_t), _bits(expected.t())), knobs


def test_masked_load_of_the_stored_tensor_matches_eager() -> None:
    """The output is the tensor the predicated packet loads (each element
    reads itself): the packets load before the lanes' stores."""
    config = {
        "block_sizes": [4, 16],
        "num_threads": [4, 2],
        "cute_vector_widths": [1, 8],
    }
    generator = torch.Generator().manual_seed(1)
    start = torch.randn((9, 32), generator=generator).to(torch.bfloat16).to(DEVICE)
    y = torch.randn((9, 32), generator=generator).to(torch.bfloat16).to(DEVICE)
    keep = (torch.arange(9, device=DEVICE) % 3) != 1
    expected = (
        torch.where(keep[:, None], start.float(), torch.zeros_like(start).float())
        + y.float()
    ).to(torch.bfloat16)
    for knobs in ("off", "guarded"):
        x = start.clone()
        _kernel(masked_accumulate, config, knobs)(x, keep, y)  # type: ignore[operator]
        assert torch.equal(_bits(x), _bits(expected)), knobs


def test_mask_reread_after_a_store_matches_eager() -> None:
    """A load masked by a flag read after the store that inverts it, stored
    under the flag read before: no element has both, so nothing is stored,
    with the lanes under the store's guard as without."""
    generator = torch.Generator().manual_seed(2)
    x = (torch.randn(37, generator=generator) + 3.0).to(DEVICE)
    start = (torch.rand(37, generator=generator) < 0.5).to(DEVICE)
    for knobs in ("off", "guarded"):
        flags = start.clone()
        kernel = _kernel(
            flags_reread_after_a_store, {"block_sizes": [8], "num_threads": [8]}, knobs
        )
        inverted, both = kernel(x, flags)  # type: ignore[operator]
        assert torch.equal(flags, ~start), knobs
        assert torch.equal(inverted, (~start).to(x.dtype)), knobs
        assert torch.equal(both, torch.zeros_like(x)), knobs


def test_extrema_order_signed_zeros_and_propagate_nans() -> None:
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 2.5],
        device=DEVICE,
    )
    x = values.repeat_interleave(len(values))
    y = values.repeat(len(values))
    config = {"block_sizes": [64]}
    eager_high, eager_low = torch.maximum(x, y), torch.minimum(x, y)
    zeros = (x == 0) & (y == 0)
    for knobs in ("off", "guarded_fmnmx"):
        kernel = helion.kernel(
            extrema.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict(
                {**config, "cute_fmnmx_extrema": knobs != "off"}
            ),
        )
        high, low = kernel(x, y)
        for actual, expected in ((high, eager_high), (low, eager_low)):
            # NaN wherever either operand is one; equal values elsewhere
            # (``-0.0 == 0.0``).
            assert torch.equal(torch.isnan(actual), torch.isnan(x) | torch.isnan(y))
            valid = ~torch.isnan(expected)
            assert torch.equal(actual[valid], expected[valid]), knobs
            # Away from a pair of zeros the bits are eager's.
            plain = valid & ~zeros
            assert torch.equal(_bits(actual)[plain], _bits(expected)[plain]), knobs
        if knobs == "guarded_fmnmx":
            # IEEE 754-2019 maximum / minimum: -0.0 orders below +0.0 in
            # either operand order.
            both = zeros & (torch.signbit(x) != torch.signbit(y))
            assert both.sum() == 2
            assert not torch.signbit(high[both]).any()
            assert torch.signbit(low[both]).all()


def test_relu_and_clamp_beside_extrema_match_eager() -> None:
    """In a kernel with an fp32 maximum the knob also turns its ``relu`` and
    ``clamp`` into the single instructions.  Their values are eager's and
    their NaNs are where the input's are, with the knob off and on; a
    ``relu`` of ``-0.0`` is ``+0.0`` either way (the zero it is compared
    with), where eager keeps the sign, so the knob changes none of their
    bits."""
    specials = [float("inf"), float("-inf"), float("nan")]
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 2.5, 7.0, -3.0, *specials], device=DEVICE
    )
    x = values.repeat_interleave(len(values))
    y = values.repeat(len(values))
    expected = (torch.relu(x), torch.clamp(x, -1.0, 6.0))
    nan = torch.isnan(x)
    zero = x == 0
    results = []
    for knob in (False, True):
        kernel = helion.kernel(
            extrema_relu_clamp.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config.from_dict(
                {"block_sizes": [64], "cute_fmnmx_extrema": knob}
            ),
        )
        _high, rectified, clamped = kernel(x, y)
        for actual, eager in zip((rectified, clamped), expected, strict=True):
            assert torch.equal(torch.isnan(actual), nan), knob
            assert torch.equal(actual[~nan], eager[~nan]), knob
            # Away from a zero the bits are eager's.
            plain = ~nan & ~zero
            assert torch.equal(_bits(actual)[plain], _bits(eager)[plain]), knob
        assert zero.sum() == 2 * len(values)
        assert not torch.signbit(rectified[zero]).any(), knob
        results.append((rectified, clamped))
    for off, on in zip(*results, strict=True):
        assert torch.equal(_bits(off)[~nan], _bits(on)[~nan])


# The layouts of the elementwise integer kernels over 256 columns: one
# element per lane behind the warp dispatcher, on one warp, on four rows of
# a warp each and at the default thread layout; then packets of four and of
# eight elements on one warp.
_ELEMENT_LAYOUTS: dict[str, dict[str, object]] = {
    "dispatcher": {"block_sizes": [4, 64], "num_threads": [0, 32]},
    "one_warp": {"block_sizes": [1, 128], "num_threads": [1, 32]},
    "four_rows": {"block_sizes": [4, 128], "num_threads": [4, 32]},
    "default": {"block_sizes": [4, 64]},
    "packets_of_4": {
        "block_sizes": [1, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
    },
    "packets_of_8": {
        "block_sizes": [1, 256],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 8],
    },
}


def _integer_rows(dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Sixteen rows of 256 integers of ``dtype``: random ones over its range
    and small ones, its bounds with small offsets, alternating and alone, and
    zeros."""
    info = torch.iinfo(dtype)
    low, high = info.min, info.max
    generator = torch.Generator().manual_seed(seed)
    column = torch.arange(256)
    x = torch.randint(low, high, (16, 256), generator=generator)
    x[1] = torch.randint(-100, 100, (256,), generator=generator)
    x[2] = high - column % 7
    x[3] = low + column % 7
    x[4] = torch.where(column % 2 == 0, high, low)
    x[5] = high
    x[6] = low
    x[7] = 0
    return x.to(dtype).to(DEVICE)


def _chain_references(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Eager's results of ``extrema_chains``, by chain."""
    deep = -c
    for factor in range(2, 5):
        deep = torch.minimum(deep, -(c * factor))
    return {
        "nested_max": torch.maximum(torch.maximum(-a, -(a * 2)), -(a + 1)),
        "nested_min": torch.minimum(torch.minimum(-b, -(b * 2)), -(b + 1)),
        "deep_min": deep,
        "named_max": torch.maximum(torch.maximum(-d, -(d * 2)), -(d + 1)),
    }


def _view_chain_case(
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Inputs of ``extrema_view_chains`` and eager's results, by chain."""
    x = _integer_rows(dtype, 4)
    y = _integer_rows(dtype, 5).t().contiguous()
    a = _integer_rows(dtype, 6)[:, 0].contiguous()
    c = _integer_rows(dtype, 7)[0].contiguous()
    expected = {
        "broadcast": torch.maximum(torch.maximum(-a, -(a * 2))[:, None], -c[None, :]),
        "expanded": torch.minimum(torch.minimum(-c, -(c * 2))[None, :], -a[:, None]),
        "sliced": torch.maximum(torch.maximum(-x, -(x * 2)), -(x + 1)),
        "permuted": torch.minimum(torch.minimum(-y, -(y * 2)).permute(1, 0), -x),
        "copied": torch.minimum(torch.minimum(-x, -(x * 3)), -(x - 1)),
        "raised": torch.maximum(torch.maximum(-x, -(x * 3)), -(x - 1)),
    }
    return (a, c, x, y), expected


def _identity_chain_references(
    chains: tuple[tuple[str, str], ...], x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Eager's results of an identity chain kernel over ``x``, by chain."""
    expected = {}
    for index, (name, kind) in enumerate(chains):
        op = torch.maximum if kind == "max" else torch.minimum
        expected[name] = op(op(-x, -(x * (index + 2))), -(x + 1))
    return expected


def _mismatches(
    kernel: object,
    args: tuple[object, ...],
    expected: dict[str, torch.Tensor],
    layouts: dict[str, dict[str, object]],
) -> list[tuple[str, str]]:
    """The (output, layout) pairs at which ``kernel`` differs from eager."""
    wrong = []
    for layout, config in layouts.items():
        outputs = _kernel(kernel, config, "off")(*args)  # type: ignore[operator]
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        for (name, reference), out in zip(expected.items(), outputs, strict=True):
            if not torch.equal(out, reference):
                wrong.append((name, layout))
    return wrong


@pytest.mark.parametrize("dtype", [torch.int32, torch.int16], ids=str)
def test_chains_of_integer_extrema_of_negated_values_equal_eager(
    dtype: torch.dtype,
) -> None:
    """ptxas (CUDA 13.3) loses one operand's negation when it fuses two
    int32 or int16 maxima, or two minima, into its three-operand
    instruction: the elementwise extremum of three or four negated values
    was that of a mix of negated and plain ones, nested or through a named
    value, as ``torch.maximum``, as lower ``clamp`` bounds, and accumulated
    by a serial loop, and also when only a view (a broadcast, a full slice,
    a ``permute``), a copy (a ``clone``, a ``repeat`` of one, a flip of no
    dim) or a value identity (``+ 0``, ``* 1``, ``| 0``, a rounding,
    ``-(-inner)``, a conversion to int64 and back, ...) lies between the
    two extrema or at the boundary of that loop, or, in a loop's body, a
    sum with a zero difference and a plain copy of the variable.  The links
    of such a chain compare through unsigned keys now (after that sum, the
    later one).  A ``relu`` over a maximum, also a chain, was right and
    stays so.

    The direct chains run at six thread layouts.  The chains through
    copies, views and identities, whose keyed code the CPU tests pin form
    by form, run at one layout each: the keys are the direct chains'."""
    a, b, c, d = (_integer_rows(dtype, seed) for seed in range(4))
    wrong = _mismatches(
        extrema_chains, (a, b, c, d), _chain_references(a, b, c, d), _ELEMENT_LAYOUTS
    )
    bounds = {
        "clamp_over_clamp": torch.clamp(torch.clamp(-a, min=-(a * 2)), min=-(a + 1)),
        "relu_over_max": torch.relu(torch.maximum(-a, -(a * 2))),
    }
    default = {"default": _ELEMENT_LAYOUTS["default"]}
    packets = {"packets_of_4": _ELEMENT_LAYOUTS["packets_of_4"]}
    wrong += _mismatches(extrema_bound_chains, (a,), bounds, default | packets)
    carried = {"carried_max": torch.amax(-a, dim=-1)}
    four_rows: dict[str, dict[str, object]] = {
        "four_rows_per_block": {"block_sizes": [4]}
    }
    one_row: dict[str, dict[str, object]] = {"one_row_per_block": {"block_sizes": [1]}}
    lowest = torch.iinfo(dtype).min
    wrong += _mismatches(extrema_carried_max, (a, lowest), carried, four_rows | one_row)
    chained = torch.maximum(torch.maximum(-a, -(a * 2)), -(a + 1))
    copies = {
        "taken": carried["carried_max"],
        "handed": carried["carried_max"],
        "repeated_in": carried["carried_max"],
        "repeated_out": carried["carried_max"],
        "flipped_out": carried["carried_max"],
        "chained": chained,
        "moved_last": chained[:, -1],
    }
    wrong += _mismatches(extrema_carried_max_copies, (a, lowest), copies, four_rows)
    # The broadcast views were wrong at the default layout only.
    wrong += _mismatches(extrema_view_chains, *_view_chain_case(dtype), default)
    for kernel, chains, layout in (
        (extrema_literal_identity_chains, LITERAL_IDENTITY_CHAINS, default),
        (extrema_identity_chains, IDENTITY_CHAINS, packets),
    ):
        references = _identity_chain_references(chains, a)
        wrong += _mismatches(kernel, (a,), references, layout)
    identities = {
        "rounded": carried["carried_max"],
        "raised": carried["carried_max"],
    }
    wrong += _mismatches(
        extrema_carried_max_identities, (a, lowest), identities, one_row
    )
    assert not wrong


@pytest.mark.parametrize("dtype", [torch.int32, torch.int16], ids=str)
def test_integer_extrema_outside_a_chain_equal_eager(dtype: torch.dtype) -> None:
    """A single maximum, a ``clamp`` of a negated value between constants
    and between negated values, and a maximum of a minimum keep their signed
    form: they were right at every layout, and stay so beside the keys."""
    x = _integer_rows(dtype)
    expected = {
        "single": torch.maximum(-x, -(x * 2)),
        "clamped": torch.clamp(-x, -5, 7),
        "bounded": torch.clamp(-x, min=-(x * 2), max=-(x + 1)),
        "mixed": torch.maximum(torch.minimum(-x, -(x * 2)), -(x + 1)),
    }
    assert not _mismatches(extrema_outside_chains, (x,), expected, _ELEMENT_LAYOUTS)


def test_chains_of_int8_extrema_of_negated_values_equal_eager() -> None:
    """Int8 keeps its signed chains, which ptxas compiles right today: the
    cases are here so that a toolchain that breaks them fails a test."""
    a, b, c, d = (_integer_rows(torch.int8, seed) for seed in range(4))
    two = {name: _ELEMENT_LAYOUTS[name] for name in ("default", "packets_of_4")}
    wrong = _mismatches(
        extrema_chains, (a, b, c, d), _chain_references(a, b, c, d), two
    )
    carried = {"carried_max": torch.amax(-a, dim=-1)}
    serial: dict[str, dict[str, object]] = {"four_rows_per_block": {"block_sizes": [4]}}
    wrong += _mismatches(extrema_carried_max, (a, -128), carried, serial)
    default = {"default": _ELEMENT_LAYOUTS["default"]}
    wrong += _mismatches(extrema_view_chains, *_view_chain_case(torch.int8), default)
    assert not wrong
