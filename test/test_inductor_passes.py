from __future__ import annotations

import contextlib
import functools
import pathlib
from types import SimpleNamespace
from typing import TYPE_CHECKING
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
from torch._inductor.lowering import lowerings
from torch._inductor.lowering import make_pointwise
from torch._inductor.utils import fresh_cache
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops as inductor_ops
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import DimDynamic
import torch.nn.functional as F
from torch.utils import _pytree as pytree

from helion._compat import requires_torch_version
from helion._compiler import backend_registry
from helion._compiler.cute import inductor_plan
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.inductor_matmul import MatmulFacts
from helion._testing import TestCase
from helion._testing import skipIfPallasTpu

if not requires_torch_version("2.16"):
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

import helion.inductor
from helion.inductor import config as helion_config
from helion.inductor.choices import HelionChoices
from helion.inductor.fx_passes import post_grad
from helion.inductor.fx_passes.cat_reductions import pointwise
from helion.inductor.fx_passes.cat_reductions import split_cat_reductions
from helion.inductor.fx_passes.post_grad import HelionPostGradPass
from helion.inductor.lowering.mm import _computed_in_kernel
from helion.inductor.lowering.mm import _read_past
from helion.inductor.lowering.mm import _slices_keep
from helion.inductor.lowering.mm import _zero_pad_source
from helion.inductor.support import split_k_reduction

if TYPE_CHECKING:
    from collections.abc import Iterator

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
            "helion._compiler.cute.cute_mma_dag",
            "helion._compiler.cute.cute_warp_mma_gemm",
            "helion._compiler.cute.dense_aligned_n",
            "helion._compiler.cute.inductor_plan",
            "helion._compiler.cute.stage_aligned_operands",
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


def native_mm(mat1: ir.TensorBox, mat2: ir.TensorBox) -> ir.TensorBox:
    """aten.mm lowered on the CPU as Inductor's native matmul lowers it for
    the dot of a GPU kernel (kernel/mm.py's tuned_mm): each operand read
    where it is computed, not stored first as Inductor's CPU mm stores it, and
    rounded to its dtype, ops.to_dtype(x, dtype, use_compute_types=False); a
    sum over K of their products in place of the dot. (Back to the compute
    type after the rounding, which the CPU backend drops where a product
    rather than a cast reads it.)"""

    def rounded(x: ir.TensorBox) -> ir.TensorBox:
        dtype = x.get_dtype()
        return make_pointwise(
            lambda value: inductor_ops.to_dtype(
                inductor_ops.to_dtype(value, dtype, use_compute_types=False), dtype
            )
        )(x)

    products = lowerings[aten.mul](
        lowerings[aten.unsqueeze](rounded(mat1), -1),
        lowerings[aten.unsqueeze](rounded(mat2), 0),
    )
    return lowerings[aten.sum](products, [1])


# The TPU job's compiles make no pad decisions, with HelionChoices installed or not
skip_on_tpu = skipIfPallasTpu("Helion's mm lowering makes no pad decisions (#4746)")


class TestZeroPadLowering(TestCase):
    @skip_on_tpu
    def test_dynamic_pad(self):
        # A stock compile of an mm over a zero pad reads none of the pad's FX nodes
        # (HelionChoices.may_rewrite()); where the plan may use it, a pad of a dynamic N
        # or M is no pad (_zero_pad_source() None), not a failed compile.
        sources = []

        def read(mat, operand):
            sources.append(_zero_pad_source(mat, operand))
            return sources[-1]

        for tcgen05, fn, a_size, b_size in (
            (False, lambda a, b: a @ F.pad(b, (0, 7)), (8, 16), (16, 9)),
            (True, lambda a, b: a @ F.pad(b, (0, 7)), (8, 16), (16, 9)),
            (True, lambda a, b: F.pad(a, (0, 0, 0, 7)) @ b, (9, 16), (16, 8)),
        ):
            sources.clear()
            a, b = torch.randn(a_size), torch.randn(b_size)
            torch._dynamo.reset()
            with (
                fresh_cache(),
                # (HelionChoices itself: where another plugin registers its own
                # choices, V.choices composes them and hides HelionChoices')
                V.set_choices_handler(HelionChoices()),
                patch.object(HelionChoices, "may_rewrite", return_value=True)
                if tcgen05
                else contextlib.nullcontext(),
                patch("helion.inductor.lowering.mm._zero_pad_source", read),
            ):
                result = torch.compile(fn, dynamic=True)(a, b)
            torch.testing.assert_close(result, fn(a, b))
            self.assertEqual(sources, [None, None] if tcgen05 else [])

    def test_read_past_slice(self):
        # Which readers of a slice of the padded product can read past it, into the pad
        # (_read_past()): the caller of a returned slice or view of it, an as_strided
        # (also through split's or unsafe_split's views) and a custom op; not ops that
        # write a new tensor.
        def product(a, b):
            return aten.slice.Tensor(torch.mm(a, F.pad(b, (0, 7))), 1, 0, 1001)

        def unsafe_split(x):
            return aten.unsafe_split(x, 500, 1)[0]

        a, b = torch.randn(64, 128), torch.randn(128, 1001)
        strided = ((64, 1008), (1008, 1))
        for fn, read_past in (
            (lambda a, b: product(a, b).relu(), False),
            (lambda a, b: product(a, b).t().unsqueeze(0).relu(), False),
            (lambda a, b: product(a, b).add_(1).relu(), False),
            (product, True),
            (lambda a, b: product(a, b).t().unsqueeze(0), True),
            (lambda a, b: product(a, b).add_(1), True),
            (lambda a, b: product(a, b).as_strided(*strided).relu(), True),
            (
                lambda a, b: product(a, b).t().as_strided((1008, 64), (1, 1008)).relu(),
                True,
            ),
            (
                lambda a, b: product(a, b).split(500, 1)[0].as_strided(*strided).relu(),
                True,
            ),
            (lambda a, b: unsafe_split(product(a, b)).relu(), False),
            (
                lambda a, b: unsafe_split(product(a, b)).as_strided(*strided).relu(),
                True,
            ),
            (lambda a, b: strided_copy(product(a, b), [64, 1008], [1008, 1]), True),
        ):
            gm = make_fx(fn, tracing_mode="fake")(a, b)
            (node,) = gm.graph.find_nodes(op="call_function", target=aten.slice.Tensor)
            self.assertTrue(_slices_keep(node, 1, 1001, 1008))
            self.assertEqual(_read_past(node), read_past)

    @contextlib.contextmanager
    def tcgen05_path(
        self, *, native: bool = False
    ) -> Iterator[list[tuple[bool, bool]]]:
        # Compiles on the CPU as on the tcgen05 path, for operands whose rows TMA loads
        # and a product kept as written; with native, through a stand-in of the GPU's
        # native matmul (native_mm()). Yields (exact, answer) of each
        # pad_matmul_output() call.
        decisions = []
        decide = HelionChoices.pad_matmul_output

        def spy(choices, *args, exact):
            decisions.append((exact, decide(choices, *args, exact=exact)))
            return decisions[-1][1]

        torch._dynamo.reset()
        with (
            fresh_cache(),
            V.set_choices_handler(HelionChoices()),  # as in test_dynamic_pad
            helion_config.patch(backend="cute"),
            patch.object(HelionChoices, "may_rewrite", return_value=True),
            patch.object(HelionChoices, "_rewritable", return_value=True),
            patch.object(HelionChoices, "transpose_matmul", return_value=False),
            patch.object(HelionChoices, "copy_matmul_b", return_value=False),
            patch.object(HelionChoices, "pad_matmul_output", spy),
            patch.dict(lowerings, {aten.mm: native_mm})
            if native
            else contextlib.nullcontext(),
        ):
            yield decisions

    def assert_pad(self, pad, expected):
        # The pad as eager holds it: NaN where eager's is, +0 (not -0) elsewhere
        torch.testing.assert_close(pad, expected, equal_nan=True, rtol=0, atol=0)
        self.assertFalse(torch.signbit(pad.nan_to_num(1.0)).any())

    @skip_on_tpu
    def test_returned_slice(self):
        # The caller of a returned slice can view past it, into pad columns that hold
        # eager's sum(|a| * 0, 1) (_pad_of_product()), whether AOTAutograd returns the
        # slice or an _unsafe_view of it; an M pad's rows hold sum(|b| * 0, 0).
        def n_pad(a, b):
            return torch.mm(a, F.pad(b, (0, 7)))[:, :1001]

        def m_pad(a, b):
            return torch.mm(F.pad(a, (0, 0, 0, 7)), b)[:1001]

        past = ((64, 7), (1008, 1), 1001)
        for requires_grad in (False, True):
            a = torch.randn(64, 128)
            a[3, 5] = float("inf")
            a[5] = -a[5].abs() - 1
            a.requires_grad_(requires_grad)
            # K-major, as pad_matmul_output() takes the pad's source
            b = torch.randn(1001, 128, requires_grad=requires_grad).t()
            with self.tcgen05_path() as decisions:
                result = torch.compile(n_pad)(a, b).detach()
            expected = n_pad(a, b).detach()
            self.assertEqual(decisions, [(True, True)])
            torch.testing.assert_close(result, expected, equal_nan=True)
            self.assert_pad(result.as_strided(*past), expected.as_strided(*past))
            self.assertTrue(result.as_strided(*past)[3].isnan().all())

        a, b = torch.randn(1001, 128), torch.randn(128, 64)
        b[5, 3] = float("-inf")
        with self.tcgen05_path() as decisions:
            result = torch.compile(m_pad)(a, b)
        expected = m_pad(a, b)
        self.assertEqual(decisions, [(True, True)])
        torch.testing.assert_close(result, expected, equal_nan=True)
        past = ((7, 64), (64, 1), 1001 * 64)
        self.assert_pad(result.as_strided(*past), expected.as_strided(*past))

    @skip_on_tpu
    def test_computed_operand(self):
        # An operand computed in the kernel, a prologue x * 2, is rounded to its dtype
        # in the exact pad's reduction as in the dot: an fp16 x * 2 that overflows to
        # inf gives eager's NaN in its pad row.
        def fn(x, b):
            return torch.mm(x * 2, F.pad(b, (0, 7)))[:, :1001]

        x = torch.ones(64, 128, dtype=torch.float16)
        x[3, 5] = 40000
        b = torch.ones(1001, 128, dtype=torch.float16).t()
        with self.tcgen05_path(native=True) as decisions:
            result = torch.compile(fn)(x, b)
        expected = fn(x, b)
        self.assertEqual(decisions, [(True, True)])
        torch.testing.assert_close(result, expected, equal_nan=True)
        past = ((64, 7), (1008, 1), 1001)
        self.assert_pad(result.as_strided(*past), expected.as_strided(*past))
        self.assertTrue(result.as_strided(*past)[3].isnan().all())

    def test_computed_in_kernel(self):
        # The operands the exact pad's reduction rounds (_computed_in_kernel()): x * 2
        # inlined or stored by a kernel the reduction may fuse with; not a graph input
        # or a view of one.
        graph = GraphLowering(torch.fx.symbolic_trace(lambda x: x + 1))
        layout = ir.FixedLayout(torch.device("cpu"), torch.float16, [64, 128], [128, 1])
        with V.set_graph_handler(graph):
            x = ir.TensorBox.create(ir.InputBuffer(name="x", layout=layout))
            stored = lowerings[aten.mul](x, 2)
            stored.realize()
            for node, computed in (
                (x, False),
                (lowerings[aten.permute](x, [1, 0]), False),
                (lowerings[aten.mul](x, 2), True),
                (lowerings[aten.permute](lowerings[aten.mul](x, 2), [1, 0]), True),
                (stored, True),
            ):
                self.assertEqual(_computed_in_kernel(node), computed)

    @skip_on_tpu
    def test_op_reads_past_slice(self):
        # A custom op may read past the slice it is handed, into the pad (_read_past()).
        def fn(a, b):
            product = torch.mm(a, F.pad(b, (0, 7)))[:, :1001]
            return strided_copy(product, [64, 1008], [1008, 1])

        a, b = torch.randn(64, 128), torch.randn(1001, 128).t()
        a[3, 5] = float("inf")
        with self.tcgen05_path() as decisions:
            result = torch.compile(fn)(a, b)
        expected = fn(a, b)
        self.assertEqual(decisions, [(True, True)])
        torch.testing.assert_close(result[:, :1001], expected[:, :1001], equal_nan=True)
        self.assert_pad(result[:, 1001:], expected[:, 1001:])
        self.assertTrue(result[3, 1001:].isnan().all())

    @skip_on_tpu
    def test_as_strided_reads_eager_layout(self):
        # An as_strided reads the slice by eager's layout of the padded product, whose
        # rows comprehensive padding (forced here on the CPU) would widen.
        def fn(a, b):
            product = torch.mm(a, F.pad(b, (0, 9)))[:, :1031]
            return product.as_strided((64, 1040), (1040, 1)) * 2

        a, b = torch.randn(64, 128), torch.randn(1031, 128).t()
        a[3, 5] = float("inf")
        with (
            config.patch(comprehensive_padding=True, disable_padding_cpu=False),
            self.tcgen05_path() as decisions,
        ):
            result = torch.compile(fn)(a, b)
        self.assertEqual(decisions, [(True, True)])
        # (a CPU BLAS may round the unpadded product differently)
        torch.testing.assert_close(
            result, fn(a, b), equal_nan=True, rtol=1e-4, atol=1e-4
        )

    @skip_on_tpu
    def test_slice_read_inside(self):
        # Where every reader reads within the slice, the pad holds zeros: no reduction
        # over a.
        def fn(a, b):
            return torch.mm(a, F.pad(b, (0, 7)))[:, :1001].relu()

        a, b = torch.randn(64, 128), torch.randn(1001, 128).t()
        a[3, 5] = float("inf")
        with self.tcgen05_path() as decisions:
            result = torch.compile(fn)(a, b)
        self.assertEqual(decisions, [(False, True)])
        torch.testing.assert_close(result, fn(a, b), equal_nan=True)

    @skip_on_tpu
    def test_source_written_before_the_product(self):
        # A write into the pad's source between the pad and the product: left in
        # the input's copy_ epilogue, the dot reads the source as the pad did;
        # made in place before the product (reinplace_inplaceable_ops), into the
        # source or through an int32 view aliasing it, the rule declines.
        def fn(a, b, idx, v):
            padded = F.pad(a, (0, 0, 0, 7))
            a.index_put_((idx,), v)
            return torch.mm(padded, b)[:1001]

        def read_first(a, b, idx, v):
            padded = F.pad(a, (0, 0, 0, 7))
            edge = padded[:2] * 2
            a.index_put_((idx,), v)
            return torch.mm(padded, b)[:1001], edge

        def through_view(a, b, idx, v):
            padded = F.pad(a, (0, 0, 0, 7))
            a.view(torch.int32).index_put_((idx,), v)
            return torch.mm(padded, b)[:1001]

        idx = torch.tensor([0, 3])
        g = torch.Generator().manual_seed(0)
        floats = (
            torch.randn(1001, 128, generator=g),
            torch.randn(128, 64, generator=g),
            idx,
            torch.randn(2, 128, generator=g),
        )
        ones = (  # writes pairs of bf16 1.0 into rows of zeros
            torch.zeros(1001, 128, dtype=torch.bfloat16),
            torch.ones(128, 64, dtype=torch.bfloat16),
            idx,
            torch.full((2, 64), 0x3F803F80, dtype=torch.int32),
        )
        no_reorder = {"reorder_for_locality": False}
        for f, (a, *rest), patches, expected in (
            (fn, floats, {}, [(True, True)]),
            (fn, floats, no_reorder, []),
            (read_first, floats, {}, []),
            (through_view, ones, no_reorder, []),
        ):
            with config.patch(patches), self.tcgen05_path() as decisions:
                result = torch.compile(f)(a.clone(), *rest)
            self.assertEqual(decisions, expected)
            torch.testing.assert_close(result, f(a.clone(), *rest))

    @skip_on_tpu
    def test_exact_pad_cost(self):
        # The exact pad's reduction reads a once, the pad kernel it saves copies b: with
        # a larger than b the rule declines, and the pad kernel's padded b gives eager's
        # pad columns.
        def fn(a, b):
            return torch.mm(a, F.pad(b, (0, 7)))[:, :9]

        a, b = torch.randn(256, 64), torch.randn(9, 64).t()
        a[3, 5] = float("inf")
        with self.tcgen05_path() as decisions:
            result = torch.compile(fn)(a, b)
        expected = fn(a, b)
        self.assertEqual(decisions, [(True, False)])
        torch.testing.assert_close(result, expected, equal_nan=True)
        past = ((256, 7), (16, 1), 9)
        self.assert_pad(result.as_strided(*past), expected.as_strided(*past))


class TestMatmulPlan(TestCase):
    def test_broadcast_operand_is_not_staged(self):
        # A K-major operand expanded along N (a 0 stride) is read as it is, as
        # the CuTe host staging leaves it: an aligned copy would write every
        # repeat.
        facts = {
            "dtype": torch.bfloat16,
            "out_dtype": torch.bfloat16,
            "tf32": False,
            "major": 10,
            "sm_count": 152,
            "m": 4096,
            "n": 131072,
            "k": 4096,
            "a_strides": (4096, 1),
            "a_offset": 0,
            "b_offset": 1,
        }
        expanded = inductor_plan.matmul_plan(MatmulFacts(**facts, b_strides=(1, 0)))
        strided = inductor_plan.matmul_plan(MatmulFacts(**facts, b_strides=(1, 4097)))
        self.assertIsNone(expanded.stage[1])
        self.assertIsNotNone(strided.stage[1])


if __name__ == "__main__":
    unittest.main()
