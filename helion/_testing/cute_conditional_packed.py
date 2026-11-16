"""Exact byte-domain observer for the production conditional SS staging helper."""

from __future__ import annotations

import ast
import importlib
from typing import TYPE_CHECKING
from typing import cast

import cutlass
import cutlass.cute as cute
import torch

from .._compiler.cute._conditional_attention_staging import stage_native_ss_pairs
from . import DEVICE
from .cuda_graph import cuda_graph_for_replay
from .cute_codegen import stage_sources
from .cute_conditional_native import validate_conditional_source

if TYPE_CHECKING:
    from pathlib import Path


def validate_packed_source(source: str) -> None:
    """Require actual packed K/V calls in each dedicated page loop."""
    validate_conditional_source(source)
    for stage in stage_sources(source):
        loops = [
            node
            for node in ast.walk(ast.parse(stage))
            if isinstance(node, ast.For)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_conditional_attention_page"
                for call in ast.walk(node)
            )
        ]
        assert len(loops) == 1
        calls = [
            call
            for call in ast.walk(loops[0])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_conditional_attention_stage_packed"
        ]
        assert len(calls) == 2
        assert ast.dump(calls[0].args[2]) != ast.dump(calls[1].args[2])


def packed_panel_inputs(phase: str) -> tuple[torch.Tensor, ...]:
    """Padded packed backing, all byte pairs, and independent runtime coordinates."""
    assert phase in ("original", "changed", "restored")
    backing = torch.full((4 * 33792,), 91, dtype=torch.int8)
    raw = backing.as_strided((4, 128, 2, 128), (33792, 256, 128, 1))
    scale = backing.view(torch.uint8).as_strided(
        (4, 128, 2, 4), (33792, 4, 512, 1), 32768
    )
    codes = torch.arange(65536, dtype=torch.int32).reshape(4, 128, 128)
    raw[:, :, 0] = (codes % 256).to(torch.uint8).view(torch.int8)
    raw[:, :, 1] = ((codes * 37 + 19) % 256).to(torch.uint8).view(torch.int8)
    scale[:, :, 0] = (codes[:, :, ::32] // 256).to(torch.uint8)
    scale[:, :, 1] = ((codes[:, :, ::32] // 256 * 73 + 11) % 256).to(torch.uint8)
    coordinates = torch.tensor(
        [(page, head, 0) for page in range(4) for head in range(2)]
        + [
            (0, 0, -127),
            (1, 1, -1),
            (2, 0, 1),
            (3, 1, 127),
            (0, 1, -128),
            (3, 0, 128),
            (-1, 0, 0),
            (4, 0, 0),
            (0, -1, 0),
            (0, 2, 0),
            (0, 0, 2147483647),
            (3, 1, -2147483648),
        ],
        dtype=torch.int64,
    )
    if phase == "changed":
        raw.copy_((raw.to(torch.int32) * 13 + 7).to(torch.int8))
        scale.copy_(((scale.to(torch.int32) * 11 + 31) % 256).to(torch.uint8))
        coordinates[:8, 0] = 3 - coordinates[:8, 0]
        coordinates[:8, 1] = 1 - coordinates[:8, 1]
        coordinates[8:14, 2].neg_()
    return backing, raw, scale, coordinates


def packed_panel_reference(
    raw: torch.Tensor, scale: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Independent Float64 BF16 lattice, typed row masks and canonical NaN bits."""
    assert raw.device.type == scale.device.type == coordinates.device.type == "cpu"
    result = torch.zeros((coordinates.shape[0], 128, 128), dtype=torch.bfloat16)
    limit = float(torch.finfo(torch.bfloat16).max)
    for task, (page, head, shift) in enumerate(coordinates.tolist()):
        if not 0 <= page < raw.shape[0] or not 0 <= head < raw.shape[2]:
            continue
        # The source's add is int32, including wrap before clamp/masking.
        rows = (torch.arange(128, dtype=torch.int64) + shift).to(torch.int32)
        visible = (rows >= 0) & (rows < 128)
        selected = rows.clamp(0, 127).long()
        expanded = scale[page, selected, head].repeat_interleave(32, -1)
        decoded = torch.ldexp(raw[page, selected, head].double(), expanded.int() - 133)
        decoded = decoded.clamp(-limit, limit).bfloat16()
        decoded.view(torch.int16)[expanded == 255] = 0x7FC0
        decoded[~visible] = 0
        result[task].copy_(decoded)
    return result


@cute.kernel
def _observe_packed_panel(
    raw: cute.Tensor, scales: cute.Tensor, coordinates: cute.Tensor, output: cute.Tensor
) -> None:
    task = cute.arch.block_idx()[0]
    thread = cute.arch.thread_idx()[0]
    storage = cute.arch.alloc_smem(cutlass.BFloat16, 16384, alignment=1024)
    panel = cute.make_tensor(storage, cute.make_layout((16384,)))
    stage_native_ss_pairs(
        raw,
        scales,
        panel,
        coordinates[task, 0],
        coordinates[task, 1],
        coordinates[task, 2],
        thread,
        (4, 128, 2, 128),
        (33792, 256, 128, 1),
        (33792, 4, 512, 1),
        cutlass.Int32,
    )
    cute.arch.sync_threads()
    # Decode the shared layout with the original scalar coordinate ownership.
    for channel in cutlass.range_constexpr(128):
        address = (
            thread * 64 + ((channel % 64) ^ ((thread % 8) * 8)) + 8192 * (channel // 64)
        )
        output[task, thread, channel] = panel[address]


@cute.jit
def _launch_packed_panel(
    raw: cute.Tensor,
    scales: cute.Tensor,
    coordinates: cute.Tensor,
    output: cute.Tensor,
    stream: object,
) -> None:
    _observe_packed_panel(raw, scales, coordinates, output).launch(
        grid=(20, 1, 1), block=(128, 1, 1), stream=stream
    )


def check_packed_panel(receipt: Path) -> None:
    """One compiled observer, original/changed/restored replay and full backing checks."""
    from cutlass.cute.runtime import from_dlpack

    cuda = importlib.import_module("cuda.bindings.driver")
    assert not receipt.exists()
    records = []
    completed = False
    try:
        with torch.cuda.device(DEVICE):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                initial, _, _, coords = packed_panel_inputs("original")
                backing = initial.to(DEVICE)
                raw = backing.as_strided((4, 128, 2, 128), (33792, 256, 128, 1))
                scales = backing.view(torch.uint8).as_strided(
                    (4, 128, 2, 4), (33792, 4, 512, 1), 32768
                )
                coordinates = coords.to(DEVICE)
                output = torch.empty(
                    (20, 128, 128), dtype=torch.bfloat16, device=DEVICE
                )
                tensors = [
                    from_dlpack(t, assumed_align=16)
                    for t in (raw, scales, coordinates, output)
                ]
                compiled = cute.compile(
                    _launch_packed_panel,
                    *tensors,
                    cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                )
                stream.synchronize()
                with cuda_graph_for_replay() as graph:
                    try:
                        with torch.cuda.graph(graph, stream=stream):
                            compiled(
                                *tensors,
                                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                            )
                        for phase in ("original", "changed", "restored"):
                            cpu_backing, cpu_raw, cpu_scale, cpu_coords = (
                                packed_panel_inputs(phase)
                            )
                            expected = packed_panel_reference(
                                cpu_raw, cpu_scale, cpu_coords
                            )
                            backing.copy_(cpu_backing)
                            coordinates.copy_(cpu_coords)
                            output.view(torch.int16).copy_(~expected.view(torch.int16))
                            graph.replay()
                            actual = output.cpu().clone().view(torch.int16)
                            after_backing = backing.cpu().clone()
                            after_coords = coordinates.cpu().clone()
                            records.append(
                                {
                                    "phase": phase,
                                    "actual": actual,
                                    "expected": expected.view(torch.int16).clone(),
                                    "input": cpu_backing,
                                    "input_after": after_backing,
                                    "coordinates": cpu_coords,
                                    "coordinates_after": after_coords,
                                }
                            )
                            torch.save(
                                {"completed": False, "records": records}, receipt
                            )
                            assert torch.equal(after_backing, cpu_backing)
                            assert torch.equal(after_coords, cpu_coords)
                            assert torch.equal(actual, expected.view(torch.int16))
                        assert torch.equal(
                            cast("torch.Tensor", records[0]["actual"]),
                            cast("torch.Tensor", records[2]["actual"]),
                        )
                        completed = True
                    finally:
                        stream.synchronize()
    finally:
        torch.save({"completed": completed, "records": records}, receipt)
