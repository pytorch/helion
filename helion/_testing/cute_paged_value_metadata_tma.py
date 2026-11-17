"""Exercise generic metadata stores followed by real TMA writes to reused slots."""

from __future__ import annotations

import importlib

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims
import torch

from .cuda_graph import cuda_graph_for_replay
from .cute_paged_value_metadata import HEADS
from .cute_paged_value_metadata import KEYS
from .cute_paged_value_metadata import PAGES
from .cute_paged_value_metadata import SENTINEL
from .cute_paged_value_metadata import STORAGE_OFFSET
from .cute_paged_value_metadata import initialize_metadata_slots
from .cute_paged_value_metadata import metadata_layout
from .cute_paged_value_metadata import observe_metadata_slot
from .cute_paged_value_metadata import produce_metadata_slots
from helion._compiler.cute._paged_attention_paired_pipeline import _enqueue_operand_tile
from helion._compiler.cute._paged_attention_paired_pipeline import _wait_non_mma
from helion._compiler.cute._paged_attention_runtime import make_paged_tma_operands

PAGE_STRIDE = 384
RAW_SENTINEL = 0x5A5A5A5A
# Each step visits both heads. Steps 0 -> 2 and 3 -> 5 overwrite copied
# metadata with TMA zero-fill, covering every ring slot and both parities.
TMA_COMMANDS = (
    (0, 0),
    (0, 1),
    (0, PAGES),
    (0, 0),
    (HEADS, 0),
    (0, PAGES),
    (0, 0),
    (0, 1),
    (0, PAGES),
)


@cute.kernel
def metadata_tma_slot_kernel(
    source: cute.Tensor,
    commands: cute.Tensor,
    observed_raw: cute.Tensor,
    observed_scales: cute.Tensor,
    observed_flags: cute.Tensor,
    raw_map: cutlass.GridConstant[cuda.TensorMap],
    scale_map: cutlass.GridConstant[cuda.TensorMap],
    read_overlap: cutlass.Constexpr,
) -> None:
    thread = cute.arch.thread_idx()[0]
    lane = cute.arch.lane_idx()
    raw = cute.arch.alloc_smem(cutlass.Int8, 4 * 16384, alignment=1024)
    scales = cute.arch.alloc_smem(cutlass.Uint8, 4 * 512, alignment=128)
    free = cute.arch.alloc_smem(cutlass.Int64, 4, alignment=8)
    metadata_ready = cute.arch.alloc_smem(cutlass.Int64, 4, alignment=8)
    raw_ready = cute.arch.alloc_smem(cutlass.Int64, 4, alignment=8)
    flags = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 4, alignment=16),
        cute.make_layout((4,)),
    )
    raw_words = cute.make_tensor(
        cute.recast_ptr(raw, dtype=cutlass.Int32), cute.make_layout((4 * 4096,))
    )
    scale_words = cute.make_tensor(
        cute.recast_ptr(scales, dtype=cutlass.Int32), cute.make_layout((4 * KEYS,))
    )
    for index in cutlass.range(thread, 4 * 4096, 96):
        raw_words[index] = cutlass.Int32(RAW_SENTINEL)
    for index in cutlass.range(thread, 4 * KEYS, 96):
        scale_words[index] = cutlass.Int32(SENTINEL)
    # These test-only initial stores also precede async TMA overwrites.
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    if thread == 0:
        initialize_metadata_slots(free, metadata_ready, raw_ready)
        prims.fence_mbarrier_init()
    cute.arch.sync_threads()
    if thread < 32:
        produce_metadata_slots(
            scales, free, flags, metadata_ready, source, commands, read_overlap
        )
    elif thread < 64:
        if lane == 0:
            for operation in range(cute.size(commands, mode=[0])):
                _enqueue_operand_tile(
                    raw,
                    scales,
                    raw_ready,
                    free,
                    raw_map,
                    scale_map,
                    cutlass.Int64(operation // 2),
                    cutlass.Int32(operation % 2),
                    cutlass.Int32(commands[operation, 1]),
                    cutlass.Int32(commands[operation, 0]),
                    metadata_ready=metadata_ready,
                    metadata_flags=flags,
                )
    else:
        raw_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((KEYS, 128), stride=(128, 1)),
        )
        for operation in range(cute.size(commands, mode=[0])):
            step = cutlass.Int64(operation // 2)
            ring = 2 * (step & 1) + operation % 2
            _wait_non_mma(raw_ready + ring, step // 2 & 1, False)
            for index in cutlass.range(lane, 4096, 32):
                offset = cute.crd2idx((index // 32, index % 32 * 4), raw_layout)
                pointer = cute.recast_ptr(
                    raw + cute.assume(ring * 16384 + offset, divby=4),
                    dtype=cutlass.Int32,
                )
                observed_raw[operation, index] = cute.make_tensor(
                    pointer, cute.make_layout((1,))
                )[0]
            observe_metadata_slot(
                scale_words,
                flags,
                free,
                observed_scales,
                observed_flags,
                operation,
                ring,
                lane,
            )
    cute.arch.sync_threads()


@cute.jit
def launch_metadata_tma_slots(
    raw: cute.Tensor,
    source: cute.Tensor,
    commands: cute.Tensor,
    observed_raw: cute.Tensor,
    observed_scales: cute.Tensor,
    observed_flags: cute.Tensor,
    stream: object,
    read_overlap: cutlass.Constexpr = False,
) -> None:
    shape, strides, _words = metadata_layout(PAGE_STRIDE)
    raw_map, scale_map = make_paged_tma_operands(
        raw,
        source,
        (128, KEYS, HEADS, 1, PAGES),
        (1, HEADS * 128, 128, KEYS * HEADS * 128, KEYS * HEADS * 128),
        shape,
        strides,
        KEYS,
    )
    metadata_tma_slot_kernel(
        source,
        commands,
        observed_raw,
        observed_scales,
        observed_flags,
        raw_map,
        scale_map,
        read_overlap,
    ).launch(grid=(1, 1, 1), block=(96, 1, 1), stream=stream)


def metadata_tma_expected(
    raw: torch.Tensor, scales: torch.Tensor, commands: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Model copied/untouched/TMA-zero-filled bytes independently of the kernel."""
    raw_slots = torch.full((4, 4096), RAW_SENTINEL, dtype=torch.int32)
    scale_slots = torch.full((4, KEYS), SENTINEL, dtype=torch.int32)
    expected_raw, expected_scales, expected_flags = [], [], []
    for operation, (first_head, physical) in enumerate(commands.tolist()):
        head = first_head + operation % 2
        ring = 2 * ((operation // 2) & 1) + operation % 2
        flag = 2
        if 0 <= head < HEADS and 0 <= physical < PAGES:
            tile = scales[:, head, 0, physical]
            flag = int(not bool(torch.any(tile)))
            if flag == 0:
                scale_slots[ring].copy_(tile)
                raw_slots[ring].copy_(
                    raw[physical, :, head, :].contiguous().view(torch.int32).flatten()
                )
        else:
            # Both TMA boxes are wholly out of bounds and zero-fill their slots.
            raw_slots[ring].zero_()
            scale_slots[ring].zero_()
        expected_raw.append(raw_slots[ring].clone())
        expected_scales.append(scale_slots[ring].clone())
        expected_flags.append(flag)
    return (
        torch.stack(expected_raw),
        torch.stack(expected_scales),
        torch.tensor(expected_flags, dtype=torch.int32),
    )


def check_metadata_tma_slots(
    device: torch.device | str, *, read_overlap: bool = False
) -> None:
    driver = importlib.import_module("cuda.bindings.driver")
    shape, strides, words = metadata_layout(PAGE_STRIDE)
    backing = torch.empty(words, dtype=torch.int32, device=device)
    source = backing.as_strided(shape, strides, STORAGE_OFFSET)
    raw = torch.empty((PAGES, KEYS, HEADS, 128), dtype=torch.int8, device=device)
    host_commands = torch.tensor(
        [command for command in TMA_COMMANDS for _head in range(2)], dtype=torch.int32
    )
    commands = host_commands.to(device)
    count = len(host_commands)
    observed = (
        torch.empty((count, 4096), dtype=torch.int32, device=device),
        torch.empty((count, KEYS), dtype=torch.int32, device=device),
        torch.empty(count, dtype=torch.int32, device=device),
    )
    arguments = [
        from_dlpack(value, assumed_align=16)
        for value in (raw, source, commands, *observed)
    ]
    compiled = cute.compile(
        launch_metadata_tma_slots,
        *arguments,
        driver.CUstream(torch.cuda.current_stream().cuda_stream),
        read_overlap=read_overlap,
    )

    def invoke() -> None:
        compiled(*arguments, driver.CUstream(torch.cuda.current_stream().cuda_stream))

    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            invoke()
        for replay in range(3):
            host_raw = (
                torch.arange(raw.numel()).reshape(raw.shape) * (17 + 2 * replay) + 11
            ).to(torch.int8)
            host_scales = (
                torch.arange(KEYS * HEADS * PAGES).reshape(shape) * (37 + 2 * replay)
                + 0x1020304
            ).to(torch.int32)
            host_scales[:, :, :, 1] = 0
            expected = metadata_tma_expected(host_raw, host_scales, host_commands)
            raw.copy_(host_raw)
            source.copy_(host_scales)
            for captured in (False, True):
                for output, reference in zip(observed, expected, strict=True):
                    output.copy_(~reference)
                if captured:
                    graph.replay()
                else:
                    invoke()
                for output, reference in zip(observed, expected, strict=True):
                    assert torch.equal(output.cpu(), reference)
