# ruff: noqa: ANN201
"""Show that PTX grid IDs stay fixed across CUDA Graph replays."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _record_grid_id(output):
    grid_id = tl.inline_asm_elementwise(
        "mov.u64 $0, %gridid;",
        "=l",
        [],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    if tl.program_id(0) == 0:
        tl.store(output, grid_id)


def main() -> None:
    output = torch.zeros(1, device="cuda", dtype=torch.uint64)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            _record_grid_id[(8,)](output, num_warps=1)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _record_grid_id[(8,)](output, num_warps=1)
    graph.replay()
    torch.cuda.synchronize()
    captured_grid_id = int(output.item())
    values = []
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        values.append(int(output.item()))
    if any(value != captured_grid_id for value in values):
        raise AssertionError(
            "grid ID changed during CUDA Graph replay: "
            f"capture={captured_grid_id}, replays={values}"
        )
    print(f"stable grid ID across CUDA Graph replay: {captured_grid_id}")


if __name__ == "__main__":
    main()
