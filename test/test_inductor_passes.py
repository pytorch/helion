from __future__ import annotations

import functools
import pathlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._dynamo.source import ConstantSource
from torch._inductor import config
from torch._inductor import ir
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.utils import _pytree as pytree

from helion._compat import requires_torch_version
from helion._compiler import backend_registry
from helion._compiler.cute import inductor_plan
from helion._compiler.cute.backend import CuteBackend
from helion._testing import TestCase

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

import helion.inductor
from helion.inductor import config as helion_config
from helion.inductor.choices import HelionChoices
from helion.inductor.fx_passes import post_grad
from helion.inductor.fx_passes.cat_reductions import pointwise
from helion.inductor.fx_passes.cat_reductions import split_cat_reductions
from helion.inductor.fx_passes.post_grad import HelionPostGradPass
from helion.inductor.support import split_k_reduction

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
        # helion.inductor: a change to the codegen in codegen/, to a pass in
        # fx_passes/ or to a lowering in lowering/ misses the cache, as one to a
        # module at the top does.
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
            "lowering/mm.py",
        }
        self.assertLessEqual(expected, hashed)

    def test_covers_backend_rules(self):
        # So does a change to any registered backend's matmul rules
        # (Backend.inductor_rules_key()), as CuteBackend hashes its rule modules.
        for backend in (CuteBackend, backend_registry.get_backend_class("pallas")):
            with patch.object(backend, "inductor_rules_key", return_value="new"):
                self.assertNotEqual(
                    helion_config._hash_sources(), helion_config._source_hash
                )
        rule_modules = {module.__name__ for module in inductor_plan.RULE_MODULES}
        expected = {
            "helion._compiler.cute.cp_async_operand",
            "helion._compiler.cute.cute_warp_mma_gemm",
            "helion._compiler.cute.dense_aligned_n",
            "helion._compiler.cute.inductor_plan",
            "helion._compiler.cute.tcgen05.search_plan",
            "helion._compiler.cute.tcgen05_constants",
        }
        self.assertLessEqual(expected, rule_modules)


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

    def test_uuid_keys_the_split_k_flags(self):
        # Inductor's key lacks allow_bf16_reduced_precision_reduction's _split_k half,
        # which split_k_reduction reads; the pass's uuid holds it (for cuda_backend
        # "helion" only).
        matmul = torch.backends.cuda.matmul
        saved = (
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction_split_k,
        )
        helion_pass = HelionPostGradPass(None)
        uuids = []
        try:
            for flags in ((False, True), (False, False)):
                matmul.allow_bf16_reduced_precision_reduction = flags
                with config.patch(cuda_backend="helion"):
                    uuids.append(helion_pass.uuid())
                with config.patch(cuda_backend="triton"):
                    self.assertIsNone(helion_pass.uuid()[2])
        finally:
            matmul.allow_bf16_reduced_precision_reduction = saved
        self.assertNotEqual(uuids[0], uuids[1])


@torch.library.custom_op("helion_test_inductor_passes::strided_copy", mutates_args=())
def strided_copy(x: torch.Tensor, size: list[int], stride: list[int]) -> torch.Tensor:
    """The elements of x's storage that x.as_strided(size, stride) views,
    which can lie past x's own: a custom op that reads past its operand."""
    return x.as_strided(size, stride).clone()


@strided_copy.register_fake
def _strided_copy_fake(
    x: torch.Tensor, size: list[int], stride: list[int]
) -> torch.Tensor:
    return x.new_empty(size)


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

    def test_unmarked_views(self):
        # aten._unsafe_view and unsafe_split alias their input, though neither
        # is_view nor their schemas say so: a slice returned through
        # _unsafe_view keeps the cat, whose storage the caller may read past
        # the slice, as does one that as_strided reads past through
        # unsafe_split. (Inductor's decompositions would trace them as views:
        # the graphs are traced without them.)
        a, b = torch.randn(4, 8), torch.randn(1, 8)

        def fn(a, b):
            return torch.ops.aten._unsafe_view(torch.cat([a, b], 0)[:4], [32])

        gm = make_fx(fn, tracing_mode="fake")(a, b)
        gm.graph.eliminate_dead_code()
        self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        torch.testing.assert_close(gm(a, b).as_strided((8,), (1,), 32), b[0])

        def split_slice(a, b):
            x = torch.unsafe_split(torch.cat([a, b], 0)[:4], 4, 1)[1]
            return x.as_strided((8,), (1,), 32)

        gm = make_fx(split_slice, tracing_mode="fake")(a, b)
        self.assertFalse(split_cat_reductions(gm.graph, devices=("cpu",)))
        gm.recompile()
        torch.testing.assert_close(gm(a, b), b[0])

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

    def test_opaque_reader(self):
        # A custom op that gets the slice, directly or through a view, and a
        # higher-order op (torch.cond's subgraphs) can read the storage past
        # it, b's part of the cat, as strided_copy() and read_past() do: the
        # cat stays. An ATen op reads only the slice's elements: a copy of a
        # replaces the slice.
        def fn(reader, a, b):
            x = torch.cat([a, b], -1)
            return x.amax(-1), reader(x[:, :32])

        def read_past(s):
            return s.as_strided((4, 40), (40, 1)).clone()

        a, b = torch.randn(4, 32), torch.randn(4, 8)
        for reader in (
            lambda s: strided_copy(s, [4, 40], [40, 1]),
            lambda s: strided_copy(s.unsqueeze(0), [4, 40], [40, 1]),
            lambda s: torch.ops.higher_order.cond(
                s.sum() > 0, read_past, read_past, (s,)
            ),
        ):
            ops = self.split(functools.partial(fn, reader), a, b)
            self.assertIn("cat.default", ops)
        ops = self.split(functools.partial(fn, lambda s: s.unsqueeze(0) * 2), a, b)
        self.assertNotIn("cat.default", ops)

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
            # ... or as as_strided_scatter and as_strided_copy do.
            lambda a, b: torch.cat([a, b], 1)[:, :32].as_strided_scatter(
                b, (4, 8), (40, 1), 32
            ),
            lambda a, b: aten.as_strided_copy(
                torch.cat([a, b], 1)[:, :32], (4, 8), (40, 1), 32
            ),
            # ... or through a getitem of a view op with several results.
            lambda a, b: (
                torch.cat([a, b], 1)[:, :32]
                .split(16, 1)[1]
                .as_strided((4, 8), (40, 1), 32)
            ),
            # ... or past a scatter's result, which copies the slice's storage
            # under its strides: as_strided reads a past b's slice.
            lambda a, b: (
                torch.cat([a, b], -1)[:, 32:]
                .select_scatter(b[0] * 2, 0, 0)
                .as_strided((4, 32), (40, 1), 0)
            ),
            lambda a, b: (
                torch.cat([a, b], -1)[:, 32:]
                .slice_scatter(b[:2] * 2, 0, 0, 2)
                .as_strided((4, 32), (40, 1), 0)
            ),
            lambda a, b: (
                torch.cat([a, b], -1)[:, 32:]
                .diagonal_scatter(b[0, :4] * 2)
                .as_strided((4, 32), (40, 1), 0)
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


class TestSplitKReduction(TestCase):
    def test_follows_the_flags(self):
        # torch's three states of a dtype's flag pair, and the strictest over a kernel's
        # 16-bit dots.
        matmul = torch.backends.cuda.matmul
        saved = (
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_fp16_reduced_precision_reduction_split_k,
        )
        try:
            for flags, expected in (
                ((True, True), "output"),
                ((False, True), "accumulator"),
                ((False, False), "none"),
            ):
                matmul.allow_fp16_reduced_precision_reduction = flags
                self.assertEqual(split_k_reduction([torch.float16]), expected)
                both = [torch.bfloat16, torch.float16]
                self.assertEqual(split_k_reduction(both), expected)
                self.assertEqual(split_k_reduction([torch.bfloat16]), "output")
        finally:
            matmul.allow_fp16_reduced_precision_reduction = saved


class TestTransposeMatmul(TestCase):
    # HelionChoices.transpose_matmul() hands the skinny rule the operands' storage
    # offsets: on GB300, bf16 [128, 1024] @ [1024, 3072] over a row-major b transposes
    # with b's rows on 128-byte lines, not 8 elements into its storage.

    @staticmethod
    def layout(size, stride, offset=0):
        return ir.FixedLayout(
            torch.device("cuda", 0), torch.bfloat16, size, stride, offset
        )

    def b_layout(self, offset=0):
        """The layout of b, row-major [1024, 3072], at `offset`."""
        return self.layout([1024, 3072], [3072, 1], offset)

    def transposes(self, graph, b):
        """Whether bf16 [128, 1024] @ b, lowered in `graph`, transposes on a CPU
        stand-in of GB300."""
        a = ir.TensorBox.create(
            ir.InputBuffer(name="a", layout=self.layout([128, 1024], [1024, 1]))
        )
        gb300 = SimpleNamespace(major=10, multi_processor_count=152)
        with (
            V.set_graph_handler(graph),
            config.patch(cuda_backend="helion"),
            helion_config.patch(backend="cute"),
            # (it reads the graph being lowered: python_wrapper())
            patch.object(HelionChoices, "use_native_matmul", return_value=True),
            patch(
                "torch._inductor.runtime.hints.DeviceProperties.create",
                return_value=gb300,
            ),
        ):
            return HelionChoices().transpose_matmul(a, b)

    def test_skinny_rule_reads_the_offsets(self):
        # b sliced from a graph input: a ReinterpretView at the slice's offset
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        storage = ir.InputBuffer(name="b", layout=self.layout([64 + 1024 * 3072], [1]))
        for offset, transposes in ((0, True), (64, True), (8, False)):
            b = ir.ReinterpretView(data=storage, layout=self.b_layout(offset))
            self.assertEqual(self.transposes(graph, b), transposes, offset)

    def test_skinny_rule_reads_input_offsets(self):
        # A graph input or constant at an offset reads offset 0 in its layout; its own
        # offset counts, adds to a view's, and a symbolic one counts by its hint.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        b = ir.TensorBox.create(ir.InputBuffer(name="b", layout=self.b_layout()))
        offsets = graph.graph_input_storage_offsets
        for offset, transposes in ((0, True), (64, True), (8, False)):
            offsets["b"] = sympy.Integer(offset)
            self.assertEqual(self.transposes(graph, b), transposes, offset)
        # a view 56 elements into an input at offset 8: 64 in all
        offsets["b"] = sympy.Integer(8)
        for view_offset, transposes in ((56, True), (0, False)):
            view = ir.ReinterpretView(data=b.data, layout=self.b_layout(view_offset))
            self.assertEqual(self.transposes(graph, view), transposes, view_offset)
        for hint, transposes in ((64, True), (8, False)):
            offsets["b"] = graph.sizevars.shape_env.create_symbol(
                hint, ConstantSource(f"b_offset_{hint}"), DimDynamic.DYNAMIC
            )
            self.assertEqual(self.transposes(graph, b), transposes, hint)
        # (only the constant's storage offset is read)
        storage = torch.empty(64 + 1024 * 3072, dtype=torch.bfloat16)
        c = ir.TensorBox.create(ir.ConstantBuffer(name="c", layout=self.b_layout()))
        for offset, transposes in ((64, True), (8, False)):
            graph.constants["c"] = storage[offset:][: 1024 * 3072].view(1024, 3072)
            self.assertEqual(self.transposes(graph, c), transposes, offset)


if __name__ == "__main__":
    unittest.main()
