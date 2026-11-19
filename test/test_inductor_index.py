from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest
import sympy
import torch
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges

from helion._compat import requires_torch_version
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

from helion.inductor.codegen.kernel import HelionKernel

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


class TestPersistentLoad(TestCase):
    def test_masked_load_through_layout(self):
        # In a persistent reduction a masked load through its buffer's own
        # layout lists the whole row as slice(None), as [tile_x, :] is a
        # SyntaxError in hl.load's index list (HelionKernel.index_list()).
        # Inductor's cat and pad index their inputs through Identity(), e.g.
        # F.pad(x[:32], (0, 0, 0, 32)) reads x at r0_1 + 32*Identity(x0), which
        # matches no layout and reads through a flat view, so this emits such a
        # load directly.
        layout = FixedLayout(torch.device("cuda"), torch.float32, [64, 300], [300, 1])
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        graph.name_to_buffer["buf0"] = InputBuffer(name="buf0", layout=layout)
        mask = CSEVariable("mask", ValueRanges.unknown(), torch.bool, ("tile_x",))
        interpreter = SimpleNamespace(current_node=None)
        with V.set_graph_handler(graph), V.set_interpreter_handler(interpreter):
            reduction = SIMDKernelFeatures([], sympy.Integer(64), sympy.Integer(300))
            kernel = HelionKernel(
                {"x": 64, "r0_": 300},
                features=reduction,
                override_persistent_reduction=True,
            )
            with V.set_kernel_handler(kernel):
                kernel.load_masks.append(mask)
                x, r = kernel.symbols["x"], kernel.symbols["r0_"]
                kernel.load("buf0", 300 * x + r)
        self.assertEqual(
            kernel.compute.getvalue(),
            "tmp0 = hl.load(__arg0__, [tile_x, slice(None)],"
            " extra_mask=mask.reshape([tile_x, 1]))\n",
        )


if __name__ == "__main__":
    unittest.main()
