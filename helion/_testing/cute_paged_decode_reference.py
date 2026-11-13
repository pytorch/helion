"""Full-domain byte decoder checks shared by native contract tests."""

from __future__ import annotations

import importlib

import pytest
import torch

from . import DEVICE
from .cuda_graph import cuda_graph_for_replay


def check_full_domain_decode(
    columns: int,
    scale_group: int,
    scale_row_stride: int,
    scale_swizzle: bool,
    transpose: bool,
    post_scale_log2: int = 0,
    *,
    positive_zero: bool = False,
) -> None:
    """Compare every byte/scale pair and graph replay using raw BF16 bits.

    Replay mixes exceptional and ordinary scales within warps, including
    isolated invalid lanes surrounded by valid rows. Bit comparison includes
    NaN, signed zero, and signed infinity cases.
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
        for replay in replays:
            if isinstance(replay, tuple):
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
            )
