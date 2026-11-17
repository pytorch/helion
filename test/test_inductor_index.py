from __future__ import annotations

import unittest

import pytest
import sympy
import torch
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import ModularIndexing

from helion._compat import requires_torch_version
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

from helion.inductor.kernel import HelionKernel

pytestmark = pytest.mark.cpu_only


def features(numel: int) -> SIMDKernelFeatures:
    """The features of a pointwise kernel of `numel` elements and no nodes."""
    return SIMDKernelFeatures([], sympy.Integer(numel))


class TestIndexWidening(TestCase):
    def test_sum_without_values(self):
        # 2**30*x + 2**30*y + (2**30 + 1)*tmp0 prints its terms without the
        # value first: each fits int32, but their sum, taken in int32 before the
        # int64 value joins, reaches 2**31, so the tile indices widen.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            kernel = HelionKernel({"y": 2, "x": 2}, features=features(4))
            with V.set_kernel_handler(kernel):
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=("tile_y", "tile_x")
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                x, y = kernel.symbols["x"], kernel.symbols["y"]
                index = 2**30 * x + 2**30 * y + (2**30 + 1) * tmp
                self.assertEqual(
                    kernel.index_tensor_expr(index, ["y", "x"]),
                    "1073741824*tile_x.index.to(torch.int64)"
                    " + 1073741824*tile_y.index.reshape([tile_y, 1]).to(torch.int64)"
                    " + 1073741825*tmp0",
                )

    def test_scalar_value(self):
        # A 0-d value, as in x.repeat(2**26).view(-1, 4).gather(0, i) with one
        # i, does not promote the int32 tile index it is added to: in int32,
        # (tile_x.index + 4*tmp0) % 100 wraps for tmp0 = 2**29.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            kernel = HelionKernel({"x": 4}, features=features(4))
            with V.set_kernel_handler(kernel):
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=()
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                x = kernel.symbols["x"]
                index = ModularIndexing(x + 4 * tmp, 1, 100)
                self.assertEqual(
                    kernel.index_tensor_expr(index, ["x"]),
                    "((tile_x.index.to(torch.int64) + 4*tmp0) % 100)",
                )


if __name__ == "__main__":
    unittest.main()
