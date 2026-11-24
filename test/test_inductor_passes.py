from __future__ import annotations

import functools
import pathlib
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from helion._compat import requires_torch_version
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

import helion.inductor
from helion.inductor import config as helion_config
from helion.inductor.fx_passes import post_grad
from helion.inductor.fx_passes.cat_reductions import pointwise
from helion.inductor.fx_passes.cat_reductions import split_cat_reductions
from helion.inductor.fx_passes.post_grad import HelionPostGradPass

pytestmark = pytest.mark.cpu_only


aten = torch.ops.aten


class Recorder(CustomGraphModulePass):
    def __init__(self, uuid: object = "recorder") -> None:
        self.graphs: list[torch.fx.GraphModule] = []
        self._uuid = uuid

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        self.graphs.append(gm)

    def uuid(self) -> object:
        return self._uuid


class TestSourceHash(TestCase):
    def test_covers_subpackages(self):
        # _source_hash, in Inductor's FX graph cache key, hashes every module of
        # helion.inductor: a change to the codegen in codegen/ or to a pass in
        # fx_passes/ misses the cache, as one to a module at the top does.
        package = pathlib.Path(helion.inductor.__file__).parent
        hashed = {
            path.relative_to(package).as_posix()
            for path in helion_config._source_files()
        }
        expected = {
            "codegen/kernel.py",
            "codegen/overrides.py",
            "codegen/scheduling.py",
            "fx_passes/cat_reductions.py",
            "fx_passes/online_softmax.py",
            "fx_passes/post_grad.py",
        }
        self.assertLessEqual(expected, hashed)


class TestPostGradPass(TestCase):
    def test_chain(self):
        # The pass registered before it still runs, also for cuda_backend
        # "triton", and its uuid stays in the FX graph cache key.
        inner = Recorder()
        helion_pass = HelionPostGradPass(inner)
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        with config.patch(cuda_backend="triton"):
            helion_pass(gm)
        self.assertEqual(inner.graphs, [gm])
        self.assertEqual(helion_pass.uuid()[1], "recorder")
        # An inner pass that asks to skip caching (uuid None) still does.
        self.assertIsNone(HelionPostGradPass(Recorder(uuid=None)).uuid())

    def test_cat_pass_without_online_softmax(self):
        # The cat pass runs with online softmax off, which only the softmax
        # rewrite reads.
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        graphs = []
        with (
            patch.object(post_grad, "split_cat_reductions", side_effect=graphs.append),
            config.patch(cuda_backend="helion", online_softmax=False),
        ):
            HelionPostGradPass(None)(gm)
        self.assertEqual(graphs, [gm.graph])


class TestCatReductions(TestCase):
    def split(self, fn, *args):
        """fn traced with Inductor's decompositions, after
        split_cat_reductions(): the targets of its ops, and its result on args
        against fn(*args)'s."""
        gm = make_fx(fn, select_decomp_table(), tracing_mode="fake")(*args)
        gm.graph.eliminate_dead_code()
        split_cat_reductions(gm.graph, devices=("cpu",))
        gm.graph.lint()
        gm.recompile()
        torch.testing.assert_close(gm(*args), fn(*args), rtol=1e-5, atol=1e-5)
        return [
            node.target.__name__
            for node in gm.graph.nodes
            if node.op == "call_function"
        ]

    def test_sink_softmax(self):
        # Softmax with attention sinks: each reduction over the rdims of 1001
        # is the reductions of the scores' 1000 and of the sink, combined, and
        # the slice that drops the sinks, made contiguous, is the
        # scores' part.
        def fn(scores, sinks):
            x = torch.cat([scores, sinks[:, None, None].expand(-1, 16, 1)], -1)
            x = x - aten.max.dim(x, -1, True)[0]
            return torch.softmax(x.float(), -1)[..., :-1].contiguous()

        ops = self.split(fn, torch.randn(4, 16, 32), torch.randn(4))
        self.assertNotIn("cat.default", ops)
        self.assertNotIn("slice.Tensor", ops)
        self.assertEqual(ops.count("maximum.default"), 2)

    def test_two_slices(self):
        # Several slices of the region, none of whose storage anything sees
        # (needs_cat_storage()): each is its parts' result, the scores' and the
        # sinks' of a softmax, or a cat of the parts an overlapping slice
        # spans.
        def sinks(scores, sinks):
            x = torch.cat([scores, sinks[:, None, None].expand(-1, 16, 1)], -1)
            p = torch.softmax(x, -1)
            return p[..., :-1].contiguous(), p[..., -1:].contiguous()

        ops = self.split(sinks, torch.randn(4, 16, 32), torch.randn(4))
        self.assertNotIn("cat.default", ops)

        def overlapping(a, b):
            x = torch.cat([a, b, a], -1) * 2
            return x.amax(-1), x[:, :40] + 1, x[:, 32:] + 1

        ops = self.split(overlapping, torch.randn(4, 32), torch.randn(4, 8))
        self.assertEqual(ops.count("cat.default"), 2)

    def test_operand_of_the_cats_size(self):
        # A tensor of the cat's size along the dim is read as one slice per
        # part.
        def fn(a, b, c):
            x = torch.cat([a, b], -1) + c
            return x.amax(-1), x[..., :32] * 2

        ops = self.split(fn, torch.randn(4, 32), torch.randn(4, 8), torch.randn(4, 40))
        self.assertNotIn("cat.default", ops)
        self.assertEqual(ops.count("slice.Tensor"), 2)

    def test_half_sum(self):
        # A float16 sum over the cat dim adds the parts' float32 sums, rounded
        # once: 512 values of 2048 and 64 of -16384 sum to 0.
        def fn(a, b):
            return torch.cat([a, b], -1).sum(-1)

        a = torch.full((4, 512), 2048.0, dtype=torch.float16)
        b = torch.full((4, 64), -16384.0, dtype=torch.float16)
        self.assertNotIn("cat.default", self.split(fn, a, b))

    def test_mixed_dtypes(self):
        # cat promotes a bfloat16 part to float32 and an int8 part to int64:
        # each part's ops run in that dtype (no exp in bfloat16, no product
        # wrapping in int8).
        a, s = torch.randn(4, 32), torch.randn(4, 1)
        fn = lambda a, s: torch.cat([a, s], -1).exp().sum(-1)  # noqa: E731
        self.assertNotIn("cat.default", self.split(fn, a.bfloat16(), s))
        a, b = torch.full((2, 32), 100, dtype=torch.int8), torch.zeros(2, 8).long()
        fn = lambda a, b: (torch.cat([a, b], -1) * 2).sum(-1)  # noqa: E731
        self.assertNotIn("cat.default", self.split(fn, a, b))

    def test_permuted_part(self):
        # The slice of cat([a.permute(1, 0, 2), b], -1) to a's part, and of
        # an op of it, viewed with their first two dims merged and doubled: the
        # slice's replacement is a contiguous copy, of which that is a view,
        # where a copy of the permuted part (or the op of it) would be none.
        a, b = torch.randn(3, 2, 32), torch.randn(2, 3, 8)
        for op in (lambda x: x, torch.exp):
            fn = functools.partial(self.permuted_part, op)
            self.assertNotIn("cat.default", self.split(fn, a, b))

    @staticmethod
    def permuted_part(op, a, b):
        x = op(torch.cat([a.permute(1, 0, 2), b], -1))
        return x[..., :32].reshape(6, 32) * 2

    def test_slice_of_another_layout(self):
        # A slice of a value that is not contiguous keeps the cat: the slice's
        # contiguous copy would not have its strides, of which the views below
        # are views. A cat of channels-last parts is channels-last, and
        # c + cat for a transposed c takes c's strides.
        a = torch.randn(4, 8, 4, 4).to(memory_format=torch.channels_last)
        b = torch.randn(1, 8, 4, 4).to(memory_format=torch.channels_last)

        def fn(a, b):
            return torch.cat([a, b], 0)[:4].permute(0, 2, 3, 1).view(4, -1).sin()

        self.assertIn("cat.default", self.split(fn, a, b))
        a, b, c = torch.randn(4, 32), torch.randn(4, 8), torch.randn(40, 4).t()

        def fn(a, b, c):
            return (c + torch.cat([a, b], -1))[:, :32].t().view(-1).sin()

        self.assertIn("cat.default", self.split(fn, a, b, c))

    def test_unsafe_view_output(self):
        # aten._unsafe_view aliases its input, though neither is_view nor its
        # schema says so: a slice returned through it keeps the cat, whose
        # storage the caller may read past the slice. (Inductor's decompositions
        # would trace it as a view: the graph is traced without them.)
        a, b = torch.randn(4, 8), torch.randn(1, 8)

        def fn(a, b):
            return torch.ops.aten._unsafe_view(torch.cat([a, b], 0)[:4], [32])

        gm = make_fx(fn, tracing_mode="fake")(a, b)
        gm.graph.eliminate_dead_code()
        self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        torch.testing.assert_close(gm(a, b).as_strided((8,), (1,), 32), b[0])

    def test_dynamic_size(self):
        # A cat of dynamic sizes stays.
        gm = make_fx(
            lambda a, b: torch.cat([a, b], -1).sum(-1),
            select_decomp_table(),
            tracing_mode="symbolic",
        )(torch.randn(4, 32), torch.randn(4, 8))
        self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))

    def test_lower_rank_operand(self):
        # A [40] scale of the cat's last dim, beside [4, 40]: sliced along its
        # own dim.
        def fn(a, b, w):
            return (torch.cat([a, b], -1) * w).sum(-1)

        ops = self.split(fn, torch.randn(4, 32), torch.randn(4, 8), torch.randn(40))
        self.assertNotIn("cat.default", ops)

    def test_slice_of_several_parts(self):
        # A slice that keeps a and b of cat([a, b, c], -1).exp() is a cat of
        # their exps: the three-part cat goes.
        a, b, c = torch.randn(4, 32), torch.randn(4, 8), torch.randn(4, 8)

        def fn(a, b, c):
            return torch.cat([a, b, c], -1).exp()[:, :40] * 2

        gm = make_fx(fn, select_decomp_table(), tracing_mode="fake")(a, b, c)
        self.assertTrue(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        cats = [n for n in gm.graph.nodes if n.target is torch.ops.aten.cat.default]
        self.assertEqual([len(n.args[0]) for n in cats], [2])
        torch.testing.assert_close(gm(a, b, c), fn(a, b, c))

    def test_pointwise_writes_a_new_tensor(self):
        # aten.positive is tagged pointwise but returns its input: a slice of
        # positive(cat) is a view of the cat, not of a part's own tensor.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        self.assertFalse(
            pointwise(graph.call_function(torch.ops.aten.positive.default, (x,)))
        )
        self.assertTrue(
            pointwise(graph.call_function(torch.ops.aten.exp.default, (x,)))
        )

    def test_sliced_output(self):
        # A slice that is a graph output, or a view of one, keeps the cat: the
        # caller may view it past its end, into the cat's storage, as eager's
        # out.as_strided((4, 8), (40, 1), 32) reads b, where a part's or a
        # copy's storage would end at the slice.
        a, b = torch.randn(4, 32), torch.randn(4, 8)
        for fn in (
            lambda a, b: (torch.cat([a, b], -1)[:, :32], torch.cat([a, b], -1).sum(-1)),
            lambda a, b: torch.cat([a, b], -1)[:, :32].t(),
            lambda a, b: torch.cat([a.t().t(), b], -1)[:, :32].unsqueeze(0),
        ):
            gm = make_fx(fn, select_decomp_table(), tracing_mode="fake")(a, b)
            gm.graph.eliminate_dead_code()
            split_cat_reductions(gm.graph, devices=("cpu",))
            gm.recompile()
            out = pytree.tree_leaves(gm(a, b))[0]
            torch.testing.assert_close(out.as_strided((4, 8), (40, 1), 32), b)

    def test_keyword_dims(self):
        # dim and keepdim given by keyword to a float16 sum and to max.dim.
        def fn(a, b):
            x = torch.cat([a, b], -1)
            return aten.sum.dim_IntList(x, [-1], True), aten.max.dim(x, -1)[0]

        a = torch.randn(4, 32, dtype=torch.float16)
        gm = make_fx(fn, tracing_mode="fake")(a, a[:, :8])
        gm.graph.eliminate_dead_code()
        for node in gm.graph.nodes:
            if node.target is aten.sum.dim_IntList:
                node.args, node.kwargs = node.args[:1], {"dim": [-1], "keepdim": True}
            elif node.target is aten.max.dim:
                node.args, node.kwargs = node.args[:1], {"dim": -1}
        self.assertTrue(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        torch.testing.assert_close(gm(a, a[:, :8]), fn(a, a[:, :8]))

    def test_keyword_cat(self):
        # A cat whose tensors and dim are keywords, as a pass before this one
        # may write it, splits like any other.
        def fn(a, b):
            return torch.cat([a, b], -1).sum(-1)

        a, b = torch.randn(4, 32), torch.randn(4, 8)
        gm = make_fx(fn, tracing_mode="fake")(a, b)
        for node in gm.graph.nodes:
            if node.target is aten.cat.default:
                node.kwargs = {"tensors": node.args[0], "dim": node.args[1]}
                node.args = ()
        self.assertTrue(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        torch.testing.assert_close(gm(a, b), fn(a, b))

    def test_keyword_only_user(self):
        # A use whose arguments are all keywords, as a user Triton kernel's
        # (triton_kernel_wrapper_functional), keeps the cat.
        def fn(a, b):
            x = torch.cat([a, b], -1)
            return x.sum(-1), torch.cumsum(x, -1)

        a, b = torch.randn(4, 32), torch.randn(4, 8)
        gm = make_fx(fn, tracing_mode="fake")(a, b)
        for node in gm.graph.nodes:
            if node.target is aten.cumsum.default:
                node.kwargs = {"self": node.args[0], "dim": node.args[1]}
                node.args = ()
        self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))

    def test_unsplit(self):
        # A use that is neither pointwise nor a reduction over the dim, a value
        # of the region that is an output, a slice inside a part, a slice that
        # as_strided reads, that is written in place or that is returned, and
        # parts none of which is thin keep the cat.
        def output(a, b):
            x = torch.cat([a, b], -1) * 2
            return x, x.sum(-1)

        def slices(a, b):
            x = torch.cat([a, b, a], -1)
            return x[:, :32], x[:, 40:]

        def overlapping(a, b):
            x = torch.cat([a, b, a], -1)
            return x[:, :40], x[:, 32:]

        def written(a, b):
            x = torch.cat([a, b], -1)
            x[:, :32].add_(1)
            return x.sum(-1)

        a, b = torch.randn(4, 32), torch.randn(4, 8)
        for fn in (
            lambda a, b: torch.cumsum(torch.cat([a, b], -1) * 2, -1).sum(-1),
            output,
            lambda a, b: torch.cat([a, b], -1).exp()[:, :16],
            # A slice that as_strided reads past it, to b, directly or through
            # a view.
            lambda a, b: torch.cat([a, b], 1)[:, :32].as_strided((4, 8), (40, 1), 32),
            lambda a, b: (
                torch.cat([a, b], 1)[:, :32]
                .unsqueeze(0)
                .as_strided((4, 8), (40, 1), 32)
            ),
            # ... or as as_strided_scatter does.
            lambda a, b: torch.cat([a, b], 1)[:, :32].as_strided_scatter(
                b, (4, 8), (40, 1), 32
            ),
            # ... or through a getitem of a view op with several results.
            lambda a, b: (
                torch.cat([a, b], 1)[:, :32]
                .split(16, 1)[1]
                .as_strided((4, 8), (40, 1), 32)
            ),
            # Parts of 32 and 24: neither is a quarter of the dim or less.
            lambda a, b: torch.cat([a, torch.cat([b, b, b], -1)], -1).sum(-1),
            # More than four parts.
            lambda a, b: torch.cat([a, b, b, b, b], -1).sum(-1),
            # max.dim whose indices are read, a sum to another dtype, and a
            # slice with a step.
            lambda a, b: torch.cat([a, b], -1).max(-1),
            lambda a, b: torch.cat([a, b], -1).sum(-1, dtype=torch.float64),
            lambda a, b: torch.cat([a, b], -1)[:, ::2],
            # Slices returned, views of the cat's storage, disjoint or not.
            slices,
            overlapping,
            # A slice written in place, a write the cat's sum must see.
            written,
        ):
            self.assertIn("cat.default", self.split(fn, a, b))
        # A part of another rank (cat's legacy empty 1-D tensor) or empty
        # along the cat dim, which Inductor's decompositions drop from the cat: traced
        # without them, the cat stays.
        for fn in (
            lambda a, b: torch.cat([a, torch.empty(0), b], -1).sum(-1),
            lambda a, b: torch.cat([a, b[:, :0], b], -1).sum(-1),
        ):
            gm = make_fx(fn, tracing_mode="fake")(a, b)
            self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))


if __name__ == "__main__":
    unittest.main()
