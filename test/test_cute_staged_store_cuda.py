"""GPU tests: a window gather's predicated packet loads and the staged store
(``cute_staged_store``) write exactly what eager computes, on partial tiles,
tiles that are not a multiple of the packet, masked tails, offset and strided
views, empty tensors, and outputs that share a storage with the input; and
the flush writes nothing outside its tile."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute import target_facts
from helion._testing import DEVICE
from helion._testing.unfold_kernels import register_tile_config
from helion._testing.unfold_kernels import sliding_windows
from helion._testing.unfold_kernels import sliding_windows_into
from helion._testing.unfold_kernels import sliding_windows_reference

# One warp of 8-lane bf16 packets on 256 flat elements, the window's 9 slots
# a per-thread lane loop over one 16-slot tile.
_WARP = register_tile_config(threads=32, window_block=16)
# A value no input holds: an element the kernel did not write keeps it.
_SENTINEL = 12345.0


def _kernel(fn: object, config: dict[str, object]) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(config),
    )


def _is_staged(kernel: helion.Kernel, args: tuple[object, ...]) -> bool:
    bound = kernel.bind(args)
    code = bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))
    return "cute.arch.sync_threads()" in code


@pytest.mark.parametrize(
    ("shape", "dtype", "window", "config", "staged"),
    [
        # 1920 elements in tiles of 256: seven staged CTAs and a partial
        # eighth, which keeps its direct stores.
        ((3, 40, 16), torch.bfloat16, 9, _WARP, True),
        # 1024 elements: every CTA stages, no test of a partial tile.
        ((2, 32, 16), torch.bfloat16, 9, _WARP, True),
        # fp32: four-element packets, three slots in a four-slot tile.
        (
            (2, 20, 16),
            torch.float32,
            3,
            register_tile_config(threads=32, window_block=4, vec=4),
            True,
        ),
        # Scalar loads staged: one element per thread, then the slots on
        # threads too (a 32 x 4 thread block).
        (
            (3, 40, 16),
            torch.bfloat16,
            9,
            {
                "block_sizes": [64, 16],
                "num_threads": [64, 1],
                "cute_staged_store": True,
            },
            True,
        ),
        (
            (3, 40, 16),
            torch.bfloat16,
            9,
            {
                "block_sizes": [32, 16],
                "num_threads": [32, 4],
                "cute_staged_store": True,
            },
            True,
        ),
        # 12 channels: the window mask differs inside a packet, so the loads
        # stay scalar under the register tile; 1440 elements leave a partial
        # tile.
        ((3, 40, 12), torch.bfloat16, 9, _WARP, True),
        # The packet loads with their direct scalar stores.
        ((3, 40, 16), torch.bfloat16, 9, _WARP | {"cute_staged_store": False}, False),
    ],
    ids=[
        "partial_last_tile",
        "full_tiles",
        "fp32",
        "element_per_thread",
        "slots_on_threads",
        "mask_inside_a_packet",
        "direct_stores",
    ],
)
def test_windows_match_eager(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    window: int,
    config: dict[str, object],
    staged: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    pad = window // 2
    kernel = _kernel(sliding_windows.fn, config)
    assert _is_staged(kernel, (x, window, pad)) == staged
    expected = sliding_windows_reference(x.view(-1), shape[1], shape[2], window, pad)
    assert torch.equal(kernel(x, window, pad), expected)


_ELEMENT_PER_THREAD = {
    "block_sizes": [64, 16],
    "num_threads": [64, 1],
    "cute_staged_store": True,
}


@pytest.mark.parametrize(
    ("shape", "dtype", "window", "config"),
    [
        # Every CTA stages: the lanes run under the staged store's mask.
        ((2, 32, 16), torch.bfloat16, 9, _WARP),
        ((2, 32, 16), torch.bfloat16, 9, _ELEMENT_PER_THREAD),
        (
            (2, 20, 16),
            torch.float32,
            3,
            register_tile_config(threads=32, window_block=4, vec=4),
        ),
        # A partial last tile: the stores are split by CTA and the lanes
        # stay ahead of them.
        ((3, 40, 16), torch.bfloat16, 9, _WARP),
        ((3, 41, 16), torch.bfloat16, 9, _ELEMENT_PER_THREAD),
    ],
    ids=[
        "packets",
        "element_per_thread",
        "fp32",
        "packets_partial_last_tile",
        "element_per_thread_partial_last_tile",
    ],
)
def test_guarded_lanes_with_a_staged_store_match_eager(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    window: int,
    config: dict[str, object],
) -> None:
    # ``cute_guarded_lanes`` leaves the staged form in place
    # (``stage_cta_stores`` runs first) and moves the gather's lanes under
    # the mask of the store into the image.
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    pad = window // 2
    kernel = _kernel(sliding_windows.fn, config | {"cute_guarded_lanes": True})
    assert _is_staged(kernel, (x, window, pad))
    expected = sliding_windows_reference(x.view(-1), shape[1], shape[2], window, pad)
    assert torch.equal(kernel(x, window, pad), expected)


@pytest.mark.parametrize(
    ("dtype", "vec"), [(torch.float32, 4), (torch.bfloat16, 8)], ids=["fp32", "bf16"]
)
def test_two_slot_windows_match_eager_in_each_form(
    dtype: torch.dtype, vec: int
) -> None:
    # Two slots are also the two components of an output element.  With
    # ``cute_vloop_sink`` the gather loads a predicated packet per slot, into
    # the staged image or with direct stores; without it a thread loads its
    # lanes' slots element by element and stores them as one component
    # packet, 32 bytes at either dtype: scalar stores before sm_100, whose
    # widest store is 16 bytes.  1968 elements leave a partial last tile at
    # either width, and the output sits inside a buffer of sentinels.
    elements, window, pad, seq, channels = 1968, 2, 1, 41, 16
    flat = (torch.arange(elements, device=DEVICE) % 1021 - 510).to(dtype)
    expected = sliding_windows_reference(flat, seq, channels, window, pad)
    tile = register_tile_config(threads=32, window_block=2, vec=vec)
    layout = {
        key: tile[key] for key in ("block_sizes", "num_threads", "cute_vector_widths")
    }
    capability = torch.cuda.get_device_capability(DEVICE)
    packs = target_facts.widest_packet_bytes(capability) >= 32
    for config, staged, sunk in (
        (tile, True, True),
        (tile | {"cute_staged_store": False}, False, True),
        (layout, False, False),
    ):
        buffer = torch.full(
            (elements * window + 64,), _SENTINEL, device=DEVICE, dtype=dtype
        )
        out = buffer[32 : 32 + elements * window].view(elements, window)
        kernel = _kernel(sliding_windows_into.fn, config)
        args = (flat, out, seq, channels, pad)
        bound = kernel.bind(args)
        code = bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))
        assert ("cute.arch.sync_threads()" in code) == staged
        assert ("_tile_store_vals" in code) == (packs and not sunk)
        assert ("_cute_load_v16b_if(flat.iterator" in code) == sunk
        kernel(*args)
        assert torch.equal(out, expected)
        assert (buffer[:32] == _SENTINEL).all()
        assert (buffer[32 + elements * window :] == _SENTINEL).all()


def test_empty_tensors_run_with_the_knob_on() -> None:
    # No CTA is launched, so there is no tile to stage: an empty batch, and
    # an output without a window slot around which nothing is written.
    scalar = {
        "block_sizes": [64, 16],
        "num_threads": [64, 1],
        "cute_staged_store": True,
    }
    empty = torch.empty((0, 40, 16), device=DEVICE, dtype=torch.bfloat16)
    kernel = _kernel(sliding_windows.fn, scalar)
    assert not _is_staged(kernel, (empty, 9, 4))
    assert kernel(empty, 9, 4).shape == (0, 9)

    flat = torch.randn(1920, device=DEVICE, dtype=torch.bfloat16)
    buffer = torch.full((64,), _SENTINEL, device=DEVICE, dtype=torch.bfloat16)
    out = buffer[32:32].view(1920, 0)
    kernel = _kernel(sliding_windows_into.fn, _WARP)
    assert not _is_staged(kernel, (flat, out, 40, 16, 4))
    assert kernel(flat, out, 40, 16, 4).shape == (1920, 0)
    assert (buffer == _SENTINEL).all()


def _buffers(elements: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A source buffer of distinct values and an output buffer of sentinels,
    each with 64 spare elements."""
    source = torch.arange(2 * elements + 64, device=DEVICE) % 1021 - 510
    return (
        source.to(torch.bfloat16),
        torch.full(
            (elements * window + 64,), _SENTINEL, device=DEVICE, dtype=torch.bfloat16
        ),
    )


@pytest.mark.parametrize(
    ("flat_view", "out_offset", "staged"),
    [
        # The output 16 bytes into a larger buffer: staged, and the buffer
        # around it keeps its sentinels.
        (slice(0, 1920), 8, True),
        # 6 bytes in: no packet may start a tile there, the stores stay
        # direct.
        (slice(0, 1920), 3, False),
        # The input 6 bytes into its buffer, and every other element of it:
        # scalar loads, staged stores.
        (slice(3, 1923), 8, True),
        (slice(0, 3840, 2), 8, True),
    ],
    ids=["aligned_output", "unaligned_output", "offset_input", "strided_input"],
)
def test_views_match_eager_and_keep_their_surroundings(
    flat_view: slice, out_offset: int, staged: bool
) -> None:
    elements, window, pad, seq, channels = 1920, 9, 4, 40, 16
    source, buffer = _buffers(elements, window)
    flat = source[flat_view]
    out = buffer[out_offset : out_offset + elements * window].view(elements, window)
    kernel = _kernel(sliding_windows_into.fn, _WARP)
    args = (flat, out, seq, channels, pad)
    assert _is_staged(kernel, args) == staged
    kernel(*args)
    assert torch.equal(out, sliding_windows_reference(flat, seq, channels, window, pad))
    # Every element was written (none keeps the sentinel a window cannot
    # hold), and nothing outside the view was.
    assert not (out == _SENTINEL).any()
    assert (buffer[:out_offset] == _SENTINEL).all()
    assert (buffer[out_offset + elements * window :] == _SENTINEL).all()


def test_an_output_in_the_inputs_storage_keeps_direct_stores() -> None:
    elements, window, pad, seq, channels = 1024, 3, 1, 32, 16
    shared = torch.full(
        (elements * (window + 1) + 64,), _SENTINEL, device=DEVICE, dtype=torch.bfloat16
    )
    shared[:elements] = (torch.arange(elements, device=DEVICE) % 509 - 254).to(
        torch.bfloat16
    )
    flat = shared[:elements]
    out = shared[elements + 32 : elements + 32 + elements * window].view(
        elements, window
    )
    kernel = _kernel(
        sliding_windows_into.fn, register_tile_config(threads=32, window_block=4)
    )
    args = (flat, out, seq, channels, pad)
    # The two views are not proven apart: no packet load moves above a
    # store, no store waits for a flush.
    assert not _is_staged(kernel, args)
    expected = sliding_windows_reference(flat.clone(), seq, channels, window, pad)
    kernel(*args)
    assert torch.equal(out, expected)
    assert (shared[elements : elements + 32] == _SENTINEL).all()
