"""Full-domain byte decoder checks shared by native contract tests."""

from __future__ import annotations

import importlib

import pytest
import torch

from . import DEVICE
from .cuda_graph import cuda_graph_for_replay

# Where the replays of the packed tile's forms place scale bytes 248 to 254,
# as (key rows, scale groups) of every tile. A K warp converts 32 rows, a V
# warp one group of all 128 rows; lane ``l`` of a V warp loads rows ``4 l``
# to ``4 l + 3``.
PACKED_GUARD_PLACEMENTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    # A few rows of one K warp, in one group: that warp is mixed, and of the
    # V warps only the group's own replaces its tile.
    "rows": ((33, 40, 62), (2,)),
    # A whole K warp: its unanimous vote fails for every row; every V warp
    # holds the bytes in a quarter of its lanes.
    "warp_rows": (tuple(range(64, 96)), (0, 1, 2, 3)),
    # A whole V warp's group: every K row is mixed, one V warp replaces its
    # tile and the other three convert natively beside it.
    "warp_group": (tuple(range(128)), (1,)),
    # The whole tile.
    "tile": (tuple(range(128)), (0, 1, 2, 3)),
    # Scattered rows of the last group: another group than the one three of
    # the V warps convert.
    "other_group": (tuple(range(0, 128, 5)), (3,)),
}
# Key rows of the replays that hold one scale byte of 248 in a tile of
# in-range bytes: byte 0, 1, 2 and 3 of a V lane's four-key scale word (lanes
# 0, 0, 15 and 31), the word that lane's vote reads. Every placement above
# has a byte in the upper half of some lane's word wherever it has one, so a
# vote that read only that half of the word would pass them all.
PACKED_GUARD_SINGLE_ROWS: tuple[int, ...] = (0, 1, 62, 127)


def check_full_domain_decode(
    columns: int,
    scale_group: int,
    scale_row_stride: int,
    scale_swizzle: bool,
    transpose: bool,
    post_scale_log2: int = 0,
    *,
    positive_zero: bool = False,
    packed_tile_forms: bool = False,
) -> None:
    """Compare every byte/scale pair and graph replay using raw BF16 bits.

    Replay mixes exceptional and ordinary scales within warps, including
    isolated invalid lanes surrounded by valid rows. Bit comparison includes
    NaN, signed zero, and signed infinity cases.

    ``packed_tile_forms`` asks the conversions for the paired body's forms
    of the packed tile (``packed_scale_tile_layout``) and, on that layout,
    adds replays that place scale bytes 248 to 254 over varied raw bytes
    where those forms decide (``PACKED_GUARD_PLACEMENTS``), and replays of
    one byte of 248 alone in a tile, in each byte of a V lane's scale word
    and each scale group (``PACKED_GUARD_SINGLE_ROWS``): a V warp's vote
    must read every key row of every lane. The reference is the source's
    full-domain arithmetic for every byte, so a form that converted an
    out-of-range byte natively, or an in-range one otherwise than exactly,
    differs in bits.
    """
    assert not positive_zero or transpose
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the decoder uses SM100 tensor memory and scaled S2F6")

    cuda = importlib.import_module("cuda.bindings.driver")
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from helion._testing.cute_paged_attention_decode import launch

    tiles = (65536 + 128 * columns - 1) // (128 * columns)
    codes = torch.arange(
        tiles * 128 * columns, device=DEVICE, dtype=torch.int32
    ).reshape(tiles, 128, columns)
    raw = (codes % 256).to(torch.uint8).view(torch.int8)
    scales = (codes[:, :, ::scale_group] // 256).to(torch.uint8).contiguous()
    output = torch.empty((tiles, 128, columns // 2), device=DEVICE, dtype=torch.int32)
    args = [from_dlpack(t, assumed_align=16) for t in (raw, scales, output)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch,
        *args,
        stream,
        scale_row_stride,
        scale_swizzle,
        transpose,
        columns,
        scale_group,
        post_scale_log2,
        positive_zero,
        packed_tile_forms,
    )
    packed_tile = (
        packed_tile_forms
        and columns == 128
        and scale_group == 32
        and scale_row_stride == 4
        and not scale_swizzle
        and post_scale_log2 >= -6
    )
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        # This is the c=0 clamp boundary; negative c uses packed BF16 scaling.
        lower = max(0, -6 - post_scale_log2)
        cutoff = -14 - post_scale_log2
        replays: list[str | int | tuple[int, int, int, int]] = ["all_pairs", "changed"]
        if post_scale_log2 < -6:
            replays.extend([0, lower - 1, lower, 247, 248])
        if post_scale_log2 <= -14:
            replays.extend([cutoff, cutoff + 1, "mixed"])
        if not transpose and columns // scale_group == 4:
            replays.extend(["sparse_first", "sparse_middle", "sparse_last"])
        if post_scale_log2 <= -14:
            # One scale outside the zero domain must veto the complete-tile
            # shortcut, including first/middle/last packets. Nonzero raw bytes
            # distinguish preserved signed zero from zero-data specialization.
            # Run these last so the existing sparse checks retain varied raw data.
            veto_positions = [(0, 0), (63, 0), (127, 0)]
            # Each byte of the packed scale word must independently veto the arm.
            # Retain the original group-zero probes without adding compile configs.
            veto_positions.extend(
                (row, group)
                for group, row in enumerate((0, 63, 127), 1)
                if group < columns // scale_group
            )
            # Isolate words one and two of each lane's four-key scale packet.
            veto_positions.extend(
                (row, group)
                for group in range(columns // scale_group)
                for row in (1, 62)
            )
            replays.extend(
                (cutoff, row, group, scale)
                for row, group in veto_positions
                for scale in (cutoff + 1, lower, 248)
            )
        if transpose and post_scale_log2 < -6:
            # Keep the complete tile in the native-only arm while distinguishing
            # every group and both middle bytes of a cached four-key word.
            replays.extend(
                (lower, row, group, 247)
                for group in range(columns // scale_group)
                for row in (1, 62)
            )
            # A lone low or exceptional byte must veto the all-native tile arm,
            # including each packed scale group and distant packets.
            native_vetoes = [(0, 0), (63, 1), (127, 2), (31, 3)]
            replays.extend(
                (lower, row, group, scale)
                for row, group in native_vetoes
                if group < columns // scale_group
                for scale in (0, lower - 1, 248)
            )
        if packed_tile:
            replays.extend(f"guard_{name}" for name in PACKED_GUARD_PLACEMENTS)
            # One byte of 248 in the first tile, every other byte 127, under
            # raw bytes of -128: the full-domain product overflows to an
            # infinity where the native conversion stays finite. Of the V
            # warps only the group's own replaces its tile, on the vote of
            # that one row of one lane; the row's K warp is mixed.
            replays.extend(
                (127, row, group, 248)
                for group in range(columns // scale_group)
                for row in PACKED_GUARD_SINGLE_ROWS
            )
        for replay in replays:
            if isinstance(replay, str) and replay.startswith("guard_"):
                # Varied raw bytes under in-range scales that differ by row
                # and group, then the placement's out-of-range bytes.
                raw.copy_(
                    ((codes * 37 + (codes // 128) * 13 + 11) % 256)
                    .to(torch.uint8)
                    .view(torch.int8)
                )
                row = torch.arange(128, device=DEVICE)[None, :, None]
                group = torch.arange(4, device=DEVICE)[None, None, :]
                tile = torch.arange(tiles, device=DEVICE)[:, None, None]
                scales.copy_(
                    ((tile * 61 + row * 5 + group * 83 + 3) % 248).to(torch.uint8)
                )
                rows, groups = PACKED_GUARD_PLACEMENTS[replay.removeprefix("guard_")]
                for position, placed_row in enumerate(rows):
                    for placed_group in groups:
                        scales[:, placed_row, placed_group] = (
                            248 + (position + placed_group) % 7
                        )
            elif isinstance(replay, tuple):
                background, row, group, scale = replay
                raw.fill_(-128)
                scales.fill_(background)
                scales[0, row, group] = scale
            elif replay == "changed":
                raw.copy_(
                    ((codes * 37 + (codes // 128) * 13 + 11) % 256)
                    .to(torch.uint8)
                    .view(torch.int8)
                )
                scales.copy_(
                    (
                        (
                            codes[:, :, ::scale_group] // 256 * 73
                            + (torch.arange(128, device=DEVICE)[None, :, None] % 2)
                            * 127
                            + torch.arange(columns // scale_group, device=DEVICE)[
                                None, None, :
                            ]
                            * 17
                            + 29
                        )
                        % 256
                    ).to(torch.uint8)
                )
            elif isinstance(replay, str) and replay.startswith("sparse_"):
                # Isolate exceptional or negative-correction scales in a row.
                # For p<-6, sparse_last is a legal c=-1 lane and uses the packed
                # multiply; otherwise it is exceptional like the first two cases.
                scales.fill_(127)
                lane, group, scale = {
                    "sparse_first": (0, 0, 248),
                    "sparse_middle": (17, 2, 255),
                    "sparse_last": (31, 3, lower - 1 if lower else 255),
                }[replay]
                scales[:, lane::32, group] = scale
            elif isinstance(replay, int):
                scales.fill_(replay)
            elif replay == "mixed":
                row = torch.arange(128, device=DEVICE)[None, :, None]
                group = torch.arange(columns // scale_group, device=DEVICE)[
                    None, None, :
                ]
                scales.copy_(
                    torch.where((row + group) % 2 == 0, cutoff, lower)
                    .to(torch.uint8)
                    .expand_as(scales)
                )
            expected = (
                raw.float()
                * torch.exp2(scales.float().repeat_interleave(scale_group, -1) - 127.0)
            ).to(torch.bfloat16)
            expected = expected * (2.0**post_scale_log2)
            if positive_zero:
                expected = expected + 0.0
            if transpose:
                expected = expected.transpose(-1, -2)
            expected = expected.contiguous()
            # Complement every expected bit, including NaN payloads, so a
            # missing write cannot accidentally match a constant sentinel.
            output.copy_(~expected.view(torch.int32))
            graph.replay()
            assert torch.equal(
                output.view(torch.bfloat16).view(torch.int16),
                expected.view(torch.int16),
            ), replay
