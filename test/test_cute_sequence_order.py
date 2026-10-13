from __future__ import annotations

import pytest
import torch

from helion.runtime.cute.launcher import default_cute_launcher

cute = pytest.importorskip("cutlass.cute")
sequence_order = pytest.importorskip("helion._compiler.cute.sequence_order")
chunk_prefill_tmem = pytest.importorskip("helion._compiler.cute.chunk_prefill_tmem")


@pytest.mark.parametrize("sequences", [1, 7, 32, 33, 513])
def test_stable_sequence_order_captures_live_offsets(sequences: int) -> None:
    from helion._compiler.cute.sequence_order import SEQUENCE_ORDER_THREADS
    from helion._compiler.cute.sequence_order import stable_sequence_order

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    lengths = [(index * 7) % 13 for index in range(sequences)]
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int64
    )
    order = torch.empty((sequences,), device="cuda", dtype=torch.int32)

    def launch() -> None:
        default_cute_launcher(
            stable_sequence_order,
            ((sequences + SEQUENCE_ORDER_THREADS - 1) // SEQUENCE_ORDER_THREADS,),
            cu,
            order,
            block=(SEQUENCE_ORDER_THREADS, 1, 1),
        )

    launch()
    torch.cuda.synchronize()
    assert order.tolist() == sorted(range(sequences), key=lambda i: -lengths[i])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for phase in range(3):
        lengths = [(index * 11 + phase) % 9 for index in range(sequences)]
        cu.copy_(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int64,
            )
        )
        snapshot = cu.clone()
        order.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert order.tolist() == sorted(range(sequences), key=lambda i: -lengths[i])
        assert torch.equal(cu, snapshot)


@cute.kernel
def _inline_order(cu: cute.Tensor, order: cute.Tensor) -> None:
    thread, _, _ = cute.arch.thread_idx()
    slot, _, _ = cute.arch.block_idx()
    index = sequence_order.select_sequence_by_length(cu, slot, thread, 1024)
    if thread % 32 == 0:
        order[slot, thread // 32] = index


@cute.kernel
def _cta_order(cu: cute.Tensor, order: cute.Tensor) -> None:
    thread, _, _ = cute.arch.thread_idx()
    slot, _, _ = cute.arch.block_idx()
    index = chunk_prefill_tmem.length_ordered_sequence(cu, slot, thread)
    if thread % 32 == 0:
        order[slot, thread // 32] = index


def _check_inline_order(cute_kernel: object, sequences: int, threads: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cu = torch.empty(sequences + 1, device="cuda", dtype=torch.int64)
    order = torch.empty((sequences, threads // 32), device="cuda", dtype=torch.int32)

    def launch() -> None:
        default_cute_launcher(
            cute_kernel, (sequences,), cu, order, block=(threads, 1, 1)
        )

    cu.zero_()
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for phase in range(3):
        lengths = [(index * 11 + phase) % 9 for index in range(sequences)]
        cu.copy_(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int64,
            )
        )
        snapshot = cu.clone()
        order.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        expected = [
            [index] * (threads // 32)
            for index in sorted(range(sequences), key=lambda i: -lengths[i])
        ]
        assert order.tolist() == expected
        assert torch.equal(cu, snapshot)


@pytest.mark.parametrize("sequences", [1, 7, 32, 33, 513])
def test_inline_sequence_order_captures_live_offsets(sequences: int) -> None:
    _check_inline_order(_inline_order, sequences, 1024)


@pytest.mark.parametrize("sequences", [1, 7, 32, 33, 513])
def test_cta_sequence_order_captures_live_offsets(sequences: int) -> None:
    _check_inline_order(_cta_order, sequences, 512)
