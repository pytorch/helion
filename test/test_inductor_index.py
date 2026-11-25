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
from torch.utils._sympy.functions import FloorDiv
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


class TestFlatLane(TestCase):
    def check(self, index_of, expected, vector=8):
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            sizes = {"z0_": 128, "y": 4, "x": 160}
            kernel = HelionKernel(sizes, features=features(128 * 4 * 160))
            with V.set_kernel_handler(kernel):
                # A gathered index read from a tensor (an embedding's), [tile_y].
                value = kernel.cse.generate(
                    kernel.compute, "src", dtype=torch.int64, shape=("tile_y",)
                )
                tmp = sympy_index_symbol(str(value))
                kernel.indirect[tmp] = value
                symbols = kernel.symbols
                index = index_of(symbols["z0_"], symbols["y"], symbols["x"], tmp)
                self.assertEqual(
                    kernel.index_tensor_expr(index, list(sizes), vector=vector),
                    expected,
                )

    def test_lane_added_last(self):
        # The H/W-swapped operand of better-benchmark's sum_9683f50f2dcf, read
        # flat (at a pointwise shape): whole 16-byte vectors beside the
        # innermost tile's index, which goes last, where CuTe's flat-gather
        # match takes it (cute/flat_gather_lane.py).
        self.check(
            lambda z, y, x, tmp: (
                x + 160 * FloorDiv(y, 2) + 4480 * ModularIndexing(y, 1, 2) + 125440 * z
            ),
            "(160*(tile_y.index.reshape([tile_y, 1]) // 2)"
            " + 4480*((tile_y.index.reshape([tile_y, 1]) % 2))"
            " + 125440*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
        )

    def test_gather_by_a_loaded_index_adds_the_lane_last(self):
        # Elements gathered by a loaded index (an embedding's lookup, tmp0 of
        # [tile_y]): the rest is the value's term, whole vectors of 160.
        self.check(
            lambda z, y, x, tmp: x + 160 * tmp + 20480 * z,
            "(160*tmp0.reshape([tile_y, 1])"
            " + 20480*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
        )

    def test_rest_of_partial_vectors_keeps_the_order(self):
        # A cat's part at -4 + 364*y: no rest of whole bf16 vectors.
        self.check(
            lambda z, y, x, tmp: x - 4 + 364 * y + 2048 * z,
            "(-4) + tile_x.index + 364*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_scaled_lane_keeps_the_order(self):
        # Every other element along x: no lane term with a unit coefficient.
        self.check(
            lambda z, y, x, tmp: 2 * x + 320 * y + 2048 * z,
            "2*tile_x.index + 320*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_vector_width_decides_the_lane(self):
        # A rest of 164*y is whole 4-element (fp32) vectors but no whole 8-element
        # (bf16) ones.
        index_of = lambda z, y, x, tmp: x + 164 * y + 2048 * z  # noqa: E731
        self.check(
            index_of,
            "(164*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])) + tile_x.index",
            vector=4,
        )
        self.check(
            index_of,
            "tile_x.index + 164*tile_y.index.reshape([tile_y, 1])"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])",
        )

    def test_a_scaled_innermost_tile_moves_the_next_one_last(self):
        # The innermost tile scaled by 16 and y added once: y is the lane.
        self.check(
            lambda z, y, x, tmp: 16 * x + y + 2048 * z,
            "(16*tile_x.index + 2048*tile_z.index.reshape([tile_z, 1, 1]))"
            " + tile_y.index.reshape([tile_y, 1])",
        )

    def test_a_rest_whose_partial_sums_may_wrap_widens(self):
        # A term read from a tensor beside int32 terms whose sum fits int32
        # but whose partial sums need not: -2**31 - 8*((y + 1) // 2) leaves
        # it before 64*((y + 4) // 3) brings it back, and 16*tmp0 (int64) can
        # join between them, which the printer's order did with the lane
        # first at x < 8*((y + 1) // 2) (the index 56 of x = 0, y = 1, tmp0 =
        # 2**27 read as 4294967352). The tile indices widen to int64, so no
        # order wraps, and the lane goes last.
        self.check(
            lambda z, y, x, tmp: (
                x - 2**31 - 8 * FloorDiv(y + 1, 2) + 16 * tmp + 64 * FloorDiv(y + 4, 3)
            ),
            "(((-2147483648)"
            " + (-8)*((1 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 2)"
            " + 16*tmp0.reshape([tile_y, 1])"
            " + 64*((4 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 3))"
            " + tile_x.index.to(torch.int64)).expand(tile_z, tile_y, tile_x)",
        )

    def test_a_value_in_a_nested_sum_widens(self):
        # The same sum inside a ModularIndexing, whose dividend wraps the
        # same way before 16*tmp0 joins (56 % 257 read as 4294967352 % 257,
        # 57): the tile indices widen too. The index adds no tile at its top,
        # so no lane moves.
        self.check(
            lambda z, y, x, tmp: ModularIndexing(
                x - 2**31 - 8 * FloorDiv(y + 1, 2) + 16 * tmp + 64 * FloorDiv(y + 4, 3),
                1,
                257,
            ),
            "((((-2147483648) + tile_x.index.to(torch.int64)"
            " + (-8)*((1 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 2)"
            " + 16*tmp0.reshape([tile_y, 1])"
            " + 64*((4 + tile_y.index.reshape([tile_y, 1]).to(torch.int64)) // 3))"
            " % 257)).expand(tile_z, tile_y, tile_x)",
        )

    def test_a_rest_whose_partial_sums_fit_adds_the_lane_last(self):
        # The same kind of rest with no negative term but the constant (8*y):
        # every partial sum of the int32 terms stays in [-2**31, -2**31 +
        # 24], and the lane goes last.
        self.check(
            lambda z, y, x, tmp: x - 2**31 + 8 * y + 16 * tmp,
            "(((-2147483648) + 8*tile_y.index.reshape([tile_y, 1])"
            " + 16*tmp0.reshape([tile_y, 1])) + tile_x.index)"
            ".expand(tile_z, tile_y, tile_x)",
        )

    def test_a_persistent_rdim_is_no_lane(self):
        # A persistent reduction's r is an arange of the whole rdim, no tile
        # whose index can be a vector lane: r + 8*x has none (x is scaled),
        # x + 8*r has x.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        with V.set_graph_handler(graph):
            reduction = SIMDKernelFeatures([], sympy.Integer(64), sympy.Integer(128))
            kernel = HelionKernel(
                {"x": 64, "r0_": 128},
                features=reduction,
                override_persistent_reduction=True,
            )
            x, r = kernel.symbols["x"], kernel.symbols["r0_"]
            self.assertIsNone(kernel.flat_lane(r + 8 * x, ["x", "r0_"], 8))
            self.assertEqual(kernel.flat_lane(x + 8 * r, ["x", "r0_"], 8), (x, 8 * r))

    def test_a_repeated_lane_keeps_the_order(self):
        # The lane in the rest too (x // 4): no rest independent of the lane.
        self.check(
            lambda z, y, x, tmp: x + 8 * FloorDiv(x, 4) + 2048 * z,
            "(tile_x.index + 8*(tile_x.index // 4)"
            " + 2048*tile_z.index.reshape([tile_z, 1, 1])).expand(tile_z, tile_y, tile_x)",
        )

    def test_flat_subscript_derives_the_vector_from_the_dtype(self):
        # flat_subscript() asks for 16 bytes of the buffer's dtype: a rest of
        # 164*y is whole vectors of 4 fp32 elements, not of 8 bf16 ones.
        for dtype, expected in (
            (torch.float32, "(164*tile_y.index.reshape([tile_y, 1])) + tile_x.index"),
            (torch.bfloat16, "tile_x.index + 164*tile_y.index.reshape([tile_y, 1])"),
        ):
            graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
            with V.set_graph_handler(graph):
                sizes = {"y": 4, "x": 160}
                kernel = HelionKernel(sizes, features=features(4 * 160))
                buffer = InputBuffer(
                    name="arg0_1",
                    layout=FixedLayout(torch.device("cuda"), dtype, [656], [1]),
                )
                with V.set_kernel_handler(kernel):
                    y, x = kernel.symbols["y"], kernel.symbols["x"]
                    _, coordinates, _ = kernel.flat_subscript(
                        buffer, x + 164 * y, list(sizes)
                    )
                    self.assertEqual(coordinates, [expected], dtype)


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
