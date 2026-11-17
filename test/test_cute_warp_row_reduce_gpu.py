from __future__ import annotations

import importlib

import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_warp_row_reduce import assert_same_float32_bits
from helion._testing.cute_warp_row_reduce import reference_row_sums


@skipIfNotCUDA()
def test_exact_row_tree_and_changed_input_replay():
    cuda = importlib.import_module("cuda.bindings.driver")
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from helion._testing.cute_warp_row_reduce import launch

    generator = torch.Generator().manual_seed(9127)
    values = torch.ldexp(
        torch.randn((4, 63, 32), generator=generator),
        torch.randint(-100, 100, (4, 63, 32), generator=generator),
    )
    values[:, 0] = -0.0
    values[0, 1, :2] = torch.tensor([float("inf"), -float("inf")])
    values[1, 3, 3] = float("nan")
    device_values = values.to(DEVICE)
    output = torch.full((4, 63), float("nan"), dtype=torch.float32, device=DEVICE)
    args = [from_dlpack(t, assumed_align=16) for t in (device_values, output)]
    compiled = cute.compile(
        launch, *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        for changed in (False, True):
            if changed:
                values = values.flip(-1).neg()
                device_values.copy_(values)
            expected = reference_row_sums(values)
            # Every sentinel differs, even when the expected result is a NaN.
            output.copy_(
                torch.where(torch.isnan(expected), 0.0, float("nan")).to(DEVICE)
            )
            graph.replay()
            assert_same_float32_bits(output.cpu(), expected)
