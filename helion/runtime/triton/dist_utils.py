from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language import core


@triton.jit
def _local_copy_block(dest, source, nelems):  # noqa: ANN001, ANN202
    for offset in range(0, nelems, 256):
        offsets = offset + tl.arange(0, 256)
        mask = offsets < nelems
        values = tl.load(source + offsets, mask=mask)
        tl.store(dest + offsets, values, mask=mask)
    tl.debug_barrier()


@triton.jit
def _nvshmem_put_nbi_block(  # noqa: ANN202
    dest,  # noqa: ANN001
    source,  # noqa: ANN001
    nelems,  # noqa: ANN001
    pe,  # noqa: ANN001
    my_pe,  # noqa: ANN001
):
    """Issue a block-scoped put whose local completion is deferred to quiet."""
    tl.static_assert(dest.type == source.type)
    if pe == my_pe:
        _local_copy_block(dest, source, nelems)
    else:
        nbytes = nelems * dest.type.element_ty.itemsize
        _nvshmem_putmem_nbi_block(
            dest.to(tl.int64),
            source.to(tl.int64),
            nbytes.to(tl.int64),
            pe,
        )


@core.extern
def _nvshmem_putmem_nbi_block(  # noqa: ANN202
    dest,  # noqa: ANN001
    source,  # noqa: ANN001
    size_bytes,  # noqa: ANN001
    pe,  # noqa: ANN001
    _semantic=None,  # noqa: ANN001
):
    return core.extern_elementwise(
        "",
        "",
        [dest, source, size_bytes, pe],
        {
            (
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("int32"),
            ): ("nvshmemx_putmem_nbi_block", core.dtype("int32"))
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _nvshmem_put_signal_nbi_block(  # noqa: ANN202
    dest,  # noqa: ANN001
    source,  # noqa: ANN001
    nelems,  # noqa: ANN001
    signal,  # noqa: ANN001
    signal_value,  # noqa: ANN001
    signal_op,  # noqa: ANN001
    pe,  # noqa: ANN001
    my_pe,  # noqa: ANN001
):
    """Issue a block-scoped NBI put with an ordered completion signal."""
    tl.static_assert(dest.type == source.type)
    nbytes = nelems * dest.type.element_ty.itemsize
    if pe == my_pe:
        _local_copy_block(dest, source, nelems)
        tl.atomic_add(
            signal,
            signal_value,
            sem="release",
            scope="sys",
        )
        tl.debug_barrier()
    else:
        _nvshmem_putmem_signal_nbi_block(
            dest.to(tl.int64),
            source.to(tl.int64),
            nbytes.to(tl.int64),
            signal.to(tl.int64),
            signal_value.to(tl.uint64),
            signal_op,
            pe,
        )


@core.extern
def _nvshmem_putmem_signal_nbi_block(  # noqa: ANN202
    dest,  # noqa: ANN001
    source,  # noqa: ANN001
    size_bytes,  # noqa: ANN001
    signal,  # noqa: ANN001
    signal_value,  # noqa: ANN001
    signal_op,  # noqa: ANN001
    pe,  # noqa: ANN001
    _semantic=None,  # noqa: ANN001
):
    return core.extern_elementwise(
        "",
        "",
        [dest, source, size_bytes, signal, signal_value, signal_op, pe],
        {
            (
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("uint64"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("nvshmemx_putmem_signal_nbi_block", core.dtype("int32"))
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _publish_signal(  # noqa: ANN202
    signal,  # noqa: ANN001
    value,  # noqa: ANN001
    signal_op,  # noqa: ANN001
    pe,  # noqa: ANN001
    my_pe,  # noqa: ANN001
):
    """Publish a counted signal to a local or remote PE."""
    if pe == my_pe:
        tl.atomic_add(signal, value, sem="release", scope="sys")
    else:
        _nvshmem_signal_op(
            signal.to(tl.int64),
            value.to(tl.int64),
            signal_op,
            pe,
        )


@core.extern
def _nvshmem_signal_op(  # noqa: ANN202
    signal,  # noqa: ANN001
    value,  # noqa: ANN001
    signal_op,  # noqa: ANN001
    pe,  # noqa: ANN001
    _semantic=None,  # noqa: ANN001
):
    return core.extern_elementwise(
        "",
        "",
        [signal, value, signal_op, pe],
        {
            (
                core.dtype("int64"),
                core.dtype("int64"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("nvshmemx_signal_op", core.dtype("int32"))
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _wait_and_consume_signal(signal, count):  # noqa: ANN001, ANN202
    """Acquire and consume counted completions from CUDA or NVSHMEM atomics."""
    value = tl.atomic_cas(
        signal,
        tl.cast(0, tl.int64),
        tl.cast(0, tl.int64),
        sem="acquire",
        scope="sys",
    )
    while value < count:
        value = tl.atomic_cas(
            signal,
            tl.cast(0, tl.int64),
            tl.cast(0, tl.int64),
            sem="acquire",
            scope="sys",
        )
    tl.atomic_add(signal, -count, sem="relaxed", scope="sys")


@triton.jit
def _add_on_every_rank(  # noqa: ANN202
    buffer_ptrs,  # noqa: ANN001
    offset,  # noqa: ANN001
    world_size: tl.constexpr,
    lanes: tl.constexpr,
    sem: tl.constexpr = "release",  # pyrefly: ignore[bad-function-definition]
):
    """Add 1 at system scope to one uint64 word on every rank.

    The caller syncs the CTA first, so a release covers all of its stores.
    """
    ranks = tl.arange(0, lanes)
    bases = tl.load(
        buffer_ptrs.to(tl.pointer_type(tl.uint64)) + ranks,
        mask=ranks < world_size,
        other=0,
    )
    tl.atomic_add(
        bases.to(tl.pointer_type(tl.uint64)) + offset,
        tl.cast(1, tl.uint64),
        mask=ranks < world_size,
        sem=sem,
        scope="sys",
    )


@triton.jit
def _store_on_every_rank(  # noqa: ANN202
    buffer_ptrs,  # noqa: ANN001
    offset,  # noqa: ANN001
    value,  # noqa: ANN001
    world_size: tl.constexpr,
    lanes: tl.constexpr,
):
    """Store one tagged uint64 word, relaxed at system scope, on every rank."""
    ranks = tl.arange(0, lanes)
    bases = tl.load(
        buffer_ptrs.to(tl.pointer_type(tl.uint64)) + ranks,
        mask=ranks < world_size,
        other=0,
    )
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
        "@p st.relaxed.sys.global.u64 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,l,r",
        [
            bases.to(tl.pointer_type(tl.uint64)) + offset,
            tl.cast(value, tl.uint64) + tl.zeros([lanes], tl.uint64),
            (ranks < world_size).to(tl.int32),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hold_after_fallback(  # noqa: ANN202
    state,  # noqa: ANN001
    started,  # noqa: ANN001
    world_size: tl.constexpr,
    epoch,  # noqa: ANN001
    held,  # noqa: ANN001
):
    """If the last launch read rows of this rank's buffer that no task wrote
    (``held``), wait until every rank has started this launch, so none still
    reads them."""
    if held:
        for rank in tl.static_range(world_size):
            _wait_at_least(state + started + rank, epoch)


@triton.jit
def _sync_threads(threads: tl.constexpr):  # noqa: ANN202
    """Barrier 0 over the first ``threads`` threads, ordering their memory ops."""
    tl.inline_asm_elementwise(
        f"bar.sync 0, {threads}; mov.u32 $0, 0;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# Inlined: Triton 3.8 rejects non-inlined calls inside warp-specialized loops.
# No reductions: their full-CTA barriers deadlock on parked WS worker warps.
@triton.jit
def _scatter_cover(  # noqa: ANN202
    state,  # noqa: ANN001
    ptrs,  # noqa: ANN001
    source,  # noqa: ANN001
    epoch,  # noqa: ANN001
    records,  # noqa: ANN001
    done,  # noqa: ANN001
    cover,  # noqa: ANN001
    votes,  # noqa: ANN001
    tasks: tl.constexpr,
    lanes: tl.constexpr,
    keys: tl.constexpr,
    world_size: tl.constexpr,
    threads: tl.constexpr,
):
    """Mark the keys ``source``'s scatter tasks wrote this launch in the local
    cover table. 1 on every thread once each task recorded its rows or was
    skipped, so an unmarked key is one no task wrote."""
    parity = tl.cast(epoch & 1, tl.int64)
    tag = tl.cast(epoch, tl.uint32)
    remote = tl.load(ptrs.to(tl.pointer_type(tl.uint64)) + source)
    remote = remote.to(tl.pointer_type(tl.uint64)) + records + parity * tasks * lanes
    marks = state + done + parity * world_size * tasks + source * tasks
    # Lanes that see a missing task bump this CTA's vote word past ``seen``.
    vote = state + votes + tl.program_id(0)
    seen = tl.load(vote, volatile=True)
    _sync_threads(threads)
    block: tl.constexpr = 8 * threads
    for start in range(0, tasks * lanes, block):
        index = start + tl.arange(0, block)
        valid = index < tasks * lanes
        mark = tl.load(marks + index // lanes, mask=valid, other=0, volatile=True)
        record = tl.load(remote + index, mask=valid, other=0, volatile=True)
        live = tl.cast(mark >> 32, tl.uint32) == tag
        skipped = (mark & 0xFFFFFFFF) == 0xFFFFFFFF
        written = tl.cast(record >> 32, tl.uint32) == tag
        tl.store(vote + index * 0, seen + 1, mask=valid & ~(live & (skipped | written)))
        key = tl.cast((record >> 1) & 0x7FFFFFFF, tl.int64)
        tl.store(
            state + cover + source * keys + key,
            epoch,
            mask=valid & written & ((record & 1) != 0) & (key < keys),
        )
    # Also orders the cover stores before every thread's reads.
    _sync_threads(threads)
    return (tl.load(vote, volatile=True) == seen).to(tl.int32)


@triton.jit
def _wait_at_least(counter, target):  # noqa: ANN001, ANN202
    """Spin in every thread with system-scope acquire loads until counter >= target."""
    value = tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u64 $0, [$1];",
        "=l,l",
        [counter],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    while value < target:
        value = tl.inline_asm_elementwise(
            "ld.acquire.sys.global.u64 $0, [$1];",
            "=l,l",
            [counter],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _get_tid():  # noqa: ANN202
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():  # noqa: ANN202
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():  # noqa: ANN202
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _get_flat_bid():  # noqa: ANN202
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def _send_signal(addrs, sem: tl.constexpr) -> None:  # noqa: ANN001
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.{sem}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs, sem: tl.constexpr) -> None:  # noqa: ANN001
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.{sem}.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _symm_mem_sync_cuda(
    signal_pad_ptrs,  # noqa: ANN001
    block_id,  # noqa: ANN001
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    hasSubsequentMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """
    Synchronizes blocks with matching block_id across participating devices.

    Note: the function itself is not a system level barrier/fence. It is a
    building block for expressing different synchronization patterns.

    Pattern 0: Ensures that all writes to symm_mem buffers from previous
    kernels across all devices are visible to the current kernel:

        symm_mem_sync(..., hasPreviousMemAccess=False, hasSubsequentMemAccess=True)

    Pattern 1: Ensures that all writes to symm_mem buffers from the current
    block are visible to all remote blocks with matching blockIdx:

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
    for writing by subsequent kernels across all devices.

        symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=False)

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal
        pad, which resets to a zero-filled state after each successful
        synchronization. This design eliminates the need for incrementing a
        flag from host.
    """
    if block_id is None:
        block_id = _get_flat_bid()
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if hasPreviousMemAccess:
        tl.debug_barrier()

    if flat_tid < world_size:
        _send_signal(send_addrs, "release" if hasPreviousMemAccess else "relaxed")  # pyrefly: ignore [bad-argument-type]
        _wait_signal(wait_addrs, "acquire" if hasSubsequentMemAccess else "relaxed")  # pyrefly: ignore [bad-argument-type]

    if hasSubsequentMemAccess:
        tl.debug_barrier()


@triton.jit
def _symm_mem_sync_rocm(
    signal_pad_ptrs,  # noqa: ANN001
    block_id,  # noqa: ANN001
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
    hasSubsequentMemAccess: tl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> None:
    """
    ROCm fallback for symmetric-memory barrier synchronization.

    This avoids CUDA inline PTX asm and relies on Triton atomics.
    """
    if block_id is None:
        block_id = _get_flat_bid()
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )

    if hasPreviousMemAccess:
        tl.debug_barrier()

    for remote_rank in tl.static_range(0, world_size):
        remote_signal_pad_addr = tl.load(signal_pad_ptrs + remote_rank).to(
            tl.pointer_type(tl.uint32)
        )
        send_addr = remote_signal_pad_addr + block_id * world_size + rank
        if hasPreviousMemAccess:
            while tl.atomic_cas(send_addr, 0, 1, sem="release", scope="sys") != 0:
                pass
        else:
            while tl.atomic_cas(send_addr, 0, 1, sem="relaxed", scope="sys") != 0:
                pass
    for remote_rank in tl.static_range(0, world_size):
        wait_addr = local_signal_pad_addr + block_id * world_size + remote_rank
        if hasSubsequentMemAccess:
            while tl.atomic_cas(wait_addr, 1, 0, sem="acquire", scope="sys") != 1:
                pass
        else:
            while tl.atomic_cas(wait_addr, 1, 0, sem="relaxed", scope="sys") != 1:
                pass

    if hasSubsequentMemAccess:
        tl.debug_barrier()


SYMM_MEM_SYNC_IMPL = (
    "rocm_atomic" if torch.version.hip is not None else "cuda_inline_asm"
)
symm_mem_sync = (
    _symm_mem_sync_rocm if torch.version.hip is not None else _symm_mem_sync_cuda
)
