from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager

if TYPE_CHECKING:
    from collections.abc import Callable


def _capture(
    fn: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


@onlyBackends(["triton", "cute"])
@skipIfNotCUDA()
@skipIfRefEager("CUDA graph tests require compiled kernels")
class TestExampleCUDAGraph(TestCase):
    def test_jagged_softmax(self) -> None:
        mod = import_path(EXAMPLES_DIR / "jagged_softmax.py")
        x = torch.randn(64, 4, device=DEVICE)
        offsets = torch.tensor([0, 0, 9, 42, 43, 43, 64], device=DEVICE)
        kernel = helion.kernel(
            mod.jagged_softmax_kernel.fn,
            config=helion.Config(block_sizes=[16, 8, 16, 16]),
        )
        graph, output = _capture(lambda: kernel(x, offsets))
        for _ in range(3):
            x.normal_()
            expected = mod.reference_jagged_softmax_pytorch(x, offsets)
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)

    def test_fused_linear_jsd(self) -> None:
        mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")
        student_input = torch.randn(16, 64, device=DEVICE)
        teacher_input = torch.randn_like(student_input)
        student_weight = torch.randn(128, 64, device=DEVICE)
        teacher_weight = torch.randn_like(student_weight)
        args = (
            0.5,
            -100,
            8.0,
            student_weight,
            teacher_weight,
            student_input,
            teacher_input,
        )
        kernel = helion.kernel(
            mod.jsd_kernel.fn,
            static_shapes=True,
            config=helion.Config(block_sizes=[4]),
        )
        with patch.object(mod, "jsd_kernel", kernel):
            graph, output = _capture(lambda: mod.fused_linear_jsd_fwd(*args))
            for _ in range(3):
                student_input.normal_()
                expected = mod.fused_linear_jsd_pytorch(*args)
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-5)
