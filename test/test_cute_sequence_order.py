from __future__ import annotations

import pytest
import torch

from helion.runtime.cute.launcher import default_cute_launcher

pytest.importorskip("cutlass.cute")


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
