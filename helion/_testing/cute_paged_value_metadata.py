"""Exercise the paired metadata slot with independent producer/checker warps."""

from __future__ import annotations

import importlib

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.experimental.primitives as prims
import torch

from .cuda_graph import cuda_graph_for_replay
from helion._compiler.cute._paged_attention_paired_pipeline import (
    _prepare_value_metadata,
)
from helion._compiler.cute._paged_attention_paired_pipeline import _wait_non_mma

KEYS = 128
HEADS = 2
PAGES = 3
HEAD_STRIDE = 160
STORAGE_OFFSET = 4
HIGH_PAGE_STRIDE = 2**30 + 128
SENTINEL = 0x13579BDF
# Every entry is visited for both pair heads. Eleven steps reuse all four slots
# through both barrier parities, including normal, zero and invalid metadata.
COMMANDS = (
    (0, 0),
    (0, 1),
    (-1, 0),
    (1, 1),
    (2**31 - 1, 1),
    (0, -1),
    (0, 2),
    (0, PAGES),
    (0, 2**31 - 1),
    (0, 0),
    (0, 1),
)


def metadata_layout(page_stride: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    shape = (KEYS, HEADS, 1, PAGES)
    strides = (1, HEAD_STRIDE, 320, page_stride)
    words = STORAGE_OFFSET + (PAGES - 1) * page_stride + HEAD_STRIDE + KEYS
    return shape, strides, words


@cute.jit
def initialize_metadata_slots(
    free: cute.Pointer,
    ready: cute.Pointer,
    raw_ready: cute.Pointer | None = None,
) -> None:
    """Initialize writer/reader counts from the single setup thread."""
    for ring in cutlass.range_constexpr(4):
        prims.mbarrier_init(free + ring, 128)
        prims.mbarrier_init(ready + ring, 32)
        if cutlass.const_expr(raw_ready is not None):
            assert raw_ready is not None
            prims.mbarrier_init(raw_ready + ring, 1)


@cute.jit
def produce_metadata_slots(
    scales: cute.Pointer,
    free: cute.Pointer,
    flags: cute.Tensor,
    ready: cute.Pointer,
    source: cute.Tensor,
    commands: cute.Tensor,
    read_overlap: cutlass.Constexpr = False,
) -> None:
    """All lanes of one warp run the metadata actor in command order."""
    for operation in range(cute.size(commands, mode=[0])):
        _prepare_value_metadata(
            scales,
            free,
            flags,
            ready,
            source,
            cutlass.Int64(operation // 2),
            cutlass.Int32(operation % 2),
            cutlass.Int32(commands[operation, 1]),
            cutlass.Int32(commands[operation, 0]),
            False,
            read_overlap=read_overlap,
        )


@cute.jit
def observe_metadata_slot(
    shared_words: cute.Tensor,
    flags: cute.Tensor,
    free: cute.Pointer,
    observed_words: cute.Tensor,
    observed_flags: cute.Tensor,
    operation: cutlass.Int32,
    ring: cutlass.Int64,
    lane: cutlass.Int32,
) -> None:
    """Read the complete slot before each checker lane releases its accesses."""
    for index in cutlass.range_constexpr(4):
        key = lane * 4 + index
        observed_words[operation, key] = shared_words[ring * KEYS + key]
    if lane == 0:
        observed_flags[operation] = flags[ring]
    # One checker warp models four converter warps. Each reading lane
    # returns four arrivals; no CTA barrier serializes it with the producer.
    cute.arch.sync_warp()
    for _arrival in cutlass.range_constexpr(4):
        prims.mbarrier_arrive(free + ring)


@cute.kernel
def metadata_slot_kernel(
    source: cute.Tensor,
    commands: cute.Tensor,
    observed_words: cute.Tensor,
    observed_flags: cute.Tensor,
    read_overlap: cutlass.Constexpr,
) -> None:
    thread = cute.arch.thread_idx()[0]
    lane = cute.arch.lane_idx()
    scale_bytes = cute.arch.alloc_smem(cutlass.Uint8, 4 * 512, alignment=16)
    free = cute.arch.alloc_smem(cutlass.Int64, 4, alignment=8)
    ready = cute.arch.alloc_smem(cutlass.Int64, 4, alignment=8)
    flags = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 4, alignment=16),
        cute.make_layout((4,)),
    )
    shared_words = cute.make_tensor(
        cute.recast_ptr(scale_bytes, dtype=cutlass.Int32), cute.make_layout((512,))
    )
    for index in cutlass.range_constexpr(8):
        shared_words[thread + index * 64] = cutlass.Int32(SENTINEL)
    if thread == 0:
        initialize_metadata_slots(free, ready)
        prims.fence_mbarrier_init()
    cute.arch.sync_threads()
    if thread < 32:
        produce_metadata_slots(
            scale_bytes, free, flags, ready, source, commands, read_overlap
        )
    else:
        for operation in range(cute.size(commands, mode=[0])):
            step = cutlass.Int64(operation // 2)
            head = cutlass.Int32(operation % 2)
            ring = 2 * (step & 1) + head
            _wait_non_mma(ready + ring, step // 2 & 1, False)
            # Give an incorrect producer that skips the free wait time to
            # overwrite the slot before the checker consumes its old contents.
            prims.nanosleep(cutlass.Uint32(10_000))
            observe_metadata_slot(
                shared_words,
                flags,
                free,
                observed_words,
                observed_flags,
                operation,
                ring,
                lane,
            )
    cute.arch.sync_threads()


@cute.jit
def launch_metadata_slots(
    source: cute.Tensor,
    commands: cute.Tensor,
    observed_words: cute.Tensor,
    observed_flags: cute.Tensor,
    stream: object,
    read_overlap: cutlass.Constexpr = False,
) -> None:
    metadata_slot_kernel(
        source, commands, observed_words, observed_flags, read_overlap
    ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def check_metadata_slots(
    page_stride: int,
    device: torch.device | str,
    *,
    sweep_words: bool,
    read_overlap: bool = False,
) -> None:
    cuda = importlib.import_module("cuda.bindings.driver")
    shape, strides, words = metadata_layout(page_stride)
    # Only the accessed tiles are initialized; the high-stride case needs a
    # real word index above Int32, not a large full-storage memset or clone.
    backing = torch.empty(words * 4, dtype=torch.uint8, device=device)
    source = backing.view(torch.int32).as_strided(shape, strides, STORAGE_OFFSET)
    assert source.data_ptr() - backing.data_ptr() == 16
    assert source.data_ptr() % 16 == 0
    if page_stride == HIGH_PAGE_STRIDE:
        assert source.stride(3) < 2**31
        assert (PAGES - 1) * source.stride(3) > 2**31
        assert source[:, 0, 0, PAGES - 1].data_ptr() - source.data_ptr() > 2**33
    commands_cpu = torch.tensor(
        [command for command in COMMANDS for _head in range(2)], dtype=torch.int32
    )
    commands = commands_cpu.to(device)
    observed = torch.empty((len(commands_cpu), KEYS), dtype=torch.int32, device=device)
    flags = torch.empty(len(commands_cpu), dtype=torch.int32, device=device)
    args = [
        from_dlpack(t, assumed_align=16) for t in (source, commands, observed, flags)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch_metadata_slots, *args, stream, read_overlap=read_overlap
    )

    def invoke() -> None:
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def inputs_and_expected(
        replay: int, *, sparse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = torch.arange(KEYS, dtype=torch.int64)[:, None, None, None]
        head = torch.arange(HEADS, dtype=torch.int64)[None, :, None, None]
        page = torch.arange(PAGES, dtype=torch.int64)[None, None, None, :]
        value = key * 19 + head * 37 + page * 73 + replay * 11
        host = torch.zeros_like(value)
        for byte in range(4):
            host |= ((value + byte * 53) & 255) << (8 * byte)
        host = host.to(torch.int32)
        if sparse:
            host.zero_()
            for source_head in range(HEADS):
                for physical in range(PAGES):
                    byte = (source_head * PAGES + physical) % 4
                    host[replay, source_head, 0, physical] = 1 << (8 * byte)
        else:
            host[:, replay % HEADS, 0, replay % PAGES] = 0
        slots = torch.full((4, KEYS), SENTINEL, dtype=torch.int32)
        expected_words = []
        expected_flags = []
        for operation, (first_head, physical) in enumerate(commands_cpu.tolist()):
            pair_head = operation % 2
            encoded_head = (first_head + pair_head + 2**31) % 2**32 - 2**31
            ring = 2 * ((operation // 2) & 1) + pair_head
            flag = 2
            if 0 <= encoded_head < HEADS and 0 <= physical < PAGES:
                tile = host[:, encoded_head, 0, physical]
                flag = int(not bool(torch.any(tile)))
                if flag == 0:
                    slots[ring].copy_(tile)
            expected_words.append(slots[ring].clone())
            expected_flags.append(flag)
        return host, torch.stack(expected_words), torch.tensor(expected_flags)

    def check(expected: torch.Tensor, expected_flags: torch.Tensor) -> None:
        actual = observed.cpu()
        assert torch.equal(actual, expected)
        assert torch.equal(flags.cpu(), expected_flags.to(torch.int32))

    host, expected, expected_flags = inputs_and_expected(0)
    source.copy_(host)
    observed.fill_(-1)
    flags.fill_(-1)
    invoke()
    check(expected, expected_flags)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            invoke()
        for replay in (1, 2):
            host, expected, expected_flags = inputs_and_expected(replay)
            source.copy_(host)
            observed.fill_(-1)
            flags.fill_(-1)
            graph.replay()
            check(expected, expected_flags)
        if sweep_words:
            # Reuse this one graph for all 128 word positions. The source
            # tiles cover all four isolated bytes, so no omitted load
            # or byte of the zero vote can silently classify a tile as zero.
            for key in range(KEYS):
                host, expected, expected_flags = inputs_and_expected(key, sparse=True)
                source.copy_(host)
                observed.fill_(-1)
                flags.fill_(-1)
                graph.replay()
                check(expected, expected_flags)
