from __future__ import annotations

import functools
import importlib
import re
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen import common
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
import torch.utils._pytree as pytree

from helion._compat import requires_torch_version
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfLowVRAM
from helion._testing import skipIfNotCUDA

if not requires_torch_version("2.16"):
    # helion.inductor uses the Inductor of torch 2.16, e.g. config.strict_reduction.
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

# After the check: torch 2.13 and 2.14 do not have it.
from torch._inductor.choices import registered_inductor_choices

import helion
import helion.inductor
from helion.inductor import config as helion_config
import helion.language as hl

SOURCE_RE = re.compile(r'helion_\w+_source = """\\\n(.*?)"""', re.DOTALL)
DOT_RE = re.compile(r"torch\.b?addb?mm\(")


# Inductor lowers matmuls natively for a backend through
# InductorChoices.use_native_matmul (pytorch/pytorch#204025).
needs_native_matmul = unittest.skipUnless(
    "use_native_matmul" in vars(InductorChoices),
    "torch without InductorChoices.use_native_matmul",
)


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device=DEVICE, dtype=dtype)


@helion.kernel(autotune_effort="none", static_shapes=True, torch_compile_fusion=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def scale_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 0.1
    return out


class InductorBackendTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(config.patch(cuda_backend="helion"))
        settings = {"native_matmul": True, "autotune_effort": "none"}
        self.enterContext(helion_config.patch(settings))

    def compile(
        self, fn, *args, expected=None, tol=None, backend="triton", expect_helion=True
    ):
        """Run compiled `fn`, check it against FP32 eager, and return the Helion
        kernel sources. Every kernel is Helion's: no Triton kernel."""
        if backend == "cute" and not _has_cute_dsl():
            self.skipTest("requires CuTeDSL")
        with (
            helion_config.patch(backend=backend),
            self.assertNoLogs("torch._inductor.scheduler", level="CRITICAL"),
        ):
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        if expected is None:
            low = (torch.float16, torch.bfloat16)
            expected = fn(*(a.float() if a.dtype in low else a for a in args))
            expected = pytree.tree_map(lambda e, r: e.to(r.dtype), expected, result)
        self.assertEqual(result, expected, atol=tol, rtol=tol)
        self.code = "\n".join(codes)
        self.assertNotIn("async_compile.triton(", self.code)
        sources = SOURCE_RE.findall(self.code)
        if expect_helion:
            self.assertTrue(sources)
            self.assertNotRegex(self.code, r"extern_kernels\.\w*mm\w*\(")
        return sources

    def stock_inductor(self, fn, *args):
        """`fn`'s output from stock Inductor (cuda_backend "triton"). It reseeds
        after, so that the next compile draws Inductor's seeds from the same
        state: a dropout mask can be compared with stock Inductor's."""
        torch._dynamo.reset()
        torch.manual_seed(0)
        with config.patch(cuda_backend="triton"):
            result = torch.compile(fn, fullgraph=True)(*args)
        torch._dynamo.reset()
        torch.manual_seed(0)
        return result

    def assertUnsupported(self, names, fn, *args, backend="triton"):
        """Check that compiling `fn` fails with Unsupported: `names`."""
        if backend == "cute" and not _has_cute_dsl():
            self.skipTest("requires CuTeDSL")
        with (
            helion_config.patch(backend=backend),
            self.assertRaisesRegex(Exception, f"Unsupported: {re.escape(names)}"),
        ):
            torch.compile(fn, fullgraph=True)(*args)

    def check_bmm_epilogue(self, backend):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(torch.bmm(a, b)) + bias,
            randn(3, 40, 72),
            randn(3, 72, 136),
            randn(136),
            backend=backend,
        )
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([3, 40, 136]):", source)
        self.assertIn("tmp0 = in0[tile_z, tile_y, tile_r]", source)
        self.assertIn("tmp1 = in1[tile_z, tile_r, tile_x]", source)
        self.assertIn("acc = torch.baddbmm(acc, tmp0, tmp1)", source)
        self.assertIn("out0[tile_z, tile_y, tile_x] = ", source)


@skipIfNotCUDA()
@onlyBackends(["triton"])
@needs_native_matmul
@instantiate_parametrized_tests
class TestInductorMatmul(InductorBackendTestCase):
    def test_mm_source(self):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(a @ b + bias),
            randn(40, 72),
            randn(72, 136),
            randn(136),
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(72):
            tmp0 = in0[tile_y, tile_r]
            tmp1 = in1[tile_r, tile_x]
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = in2[tile_x].to(torch.float32)
        tmp3 = acc + tmp2
        tmp4 = torch.relu(tmp3)
        out0[tile_y, tile_x] = tmp4.to(torch.bfloat16)
""",
        )
        self.assertIn(
            "helion_fused_add_mm_relu_0 = PyCodeCache.load("
            "helion_fused_add_mm_relu_0_source).kernel",
            self.code,
        )
        self.assertIn(
            "helion_fused_add_mm_relu_0(arg0_1, arg1_1, arg2_1, buf1)", self.code
        )

    @parametrize("native_matmul", [False, True])
    def test_other_cuda_backend(self, native_matmul):
        # With cuda_backend "triton", HelionChoices decides as InductorChoices
        # does, by triton.native_matmul: extern_kernels.mm(arg0_1, arg1_1,
        # out=buf0), or a Triton kernel with tl.dot.
        with config.patch(
            {"cuda_backend": "triton", "triton.native_matmul": native_matmul}
        ):
            _, (code,) = run_and_get_code(
                torch.compile(torch.mm), randn(40, 72), randn(72, 136)
            )
        self.assertEqual("tl.dot(" in code, native_matmul)
        self.assertEqual("extern_kernels.mm(" in code, not native_matmul)
        self.assertNotIn("helion_", code)

    @parametrize("native_matmul", [True, False])
    def test_backward(self, native_matmul):
        # AOTAutograd compiles the backward graph at its first run, in the autograd
        # thread, with the config of this thread: cuda_backend "helion" and
        # autotune_effort="none" from setUp, and native_matmul. Its dot, relu's
        # backward times b.T, is
        #   helion_fused_mm_threshold_backward_0(le, tangents_1, permute, buf0)
        # and with native_matmul=False
        #   extern_kernels.mm(buf0, permute, out=buf1)
        self.enterContext(helion_config.patch(native_matmul=native_matmul))
        # cuBLAS may reduce the split-K partials of a bf16 GEMM in bf16
        # (allow_bf16_reduced_precision_reduction, on by default). On an A10G
        # it does for extern_kernels.mm(buf0, permute): a.grad[16, 16], 136
        # products that cancel to 0.0433 in FP32, came out 0.0126, off by about
        # one bf16 ulp of the partial sums (2**-5 between 4 and 8). Reduce in
        # FP32, as the Helion dot accumulates, so the check below holds on
        # every GPU.
        matmul = torch.backends.cuda.matmul
        reduction = (
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction_split_k,
        )
        matmul.allow_bf16_reduced_precision_reduction = False
        self.addCleanup(
            setattr, matmul, "allow_bf16_reduced_precision_reduction", reduction
        )
        a, b = randn(40, 72).requires_grad_(), randn(72, 136)
        out = torch.compile(lambda a, b: torch.relu(a @ b), fullgraph=True)(a, b)
        _, (code,) = run_and_get_code(lambda: out.sum().backward())
        (source,) = SOURCE_RE.findall(code)
        self.assertIn('autotune_effort="none"', source)
        self.assertEqual(bool(DOT_RE.search(source)), native_matmul)
        self.assertEqual("extern_kernels.mm(" in code, not native_matmul)
        a_ref = a.detach().float().requires_grad_()
        torch.relu(a_ref @ b.float()).sum().backward()
        self.assertEqual(a.grad, a_ref.grad.to(a.dtype), atol=2e-2, rtol=2e-2)

    def test_bmm_epilogue(self):
        self.check_bmm_epilogue("triton")

    def test_saved_bf16_output(self):
        # silu's backward reads a @ b, so the bf16 product is saved and rounded.
        fn = lambda a, b: torch.nn.functional.silu(a @ b)  # noqa: E731
        a, b = randn(40, 72), randn(72, 136).requires_grad_()
        (source,) = self.compile(fn, a, b, expected=fn(a, b), tol=2e-2)
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(72):
            tmp0 = in0[tile_y, tile_r]
            tmp1 = in1[tile_r, tile_x]
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = acc.to(torch.bfloat16)
        tmp3 = tmp2.to(torch.float32)
        out0[tile_y, tile_x] = tmp3.to(torch.bfloat16)
        tmp4 = -tmp3
        tmp5 = torch.exp(tmp4)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp3 / tmp7
        out1[tile_y, tile_x] = tmp8.to(torch.bfloat16)
""",
        )

    def test_cast_epilogue(self):
        # Integers, so that the int32 cast is exact.
        a = torch.randint(-3, 4, (40, 72), device=DEVICE, dtype=torch.float32)
        b = torch.randint(-3, 4, (72, 136), device=DEVICE, dtype=torch.float32)
        (source,) = self.compile(lambda a, b: (a @ b).to(torch.int32) + 1, a, b)
        self.assertIn("tmp2 = acc.to(torch.int32)", source)

    def test_upcast_prologue(self):
        # a + 1 is computed in fp32, so the bf16 a is read as fp32.
        fn = lambda a, b: (a + 1).float() @ b.float()  # noqa: E731
        a, b = randn(40, 72), randn(72, 136)
        (source,) = self.compile(fn, a, b, tol=1e-3)
        self.assertIn("in0[tile_y, tile_r].to(torch.float32)", source)

    def test_out_dtype(self):
        fn = lambda a, b: torch.mm(a, b, out_dtype=torch.float32)  # noqa: E731
        a, b = randn(40, 72), randn(72, 136)
        (source,) = self.compile(fn, a, b, tol=1e-3)
        self.assertIn("out0[tile_y, tile_x] = acc\n", source)

    def test_operand_view_of_dynamic_input(self):
        x = randn(4, 40, 72)
        torch._dynamo.mark_dynamic(x, 0)
        # x[0] is a static [40, 72] view of the [s, 40, 72] input.
        (source,) = self.compile(lambda x, w: x[0] @ w, x, randn(72, 136))
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertRegex(
            self.code, r"reinterpret_tensor\(arg\d+_1, \(40, 72\), \(72, 1\), 0\)"
        )

    @config.patch(nan_asserts=True)
    def test_nan_asserts_check_the_tensor(self):
        base = randn(5, 9, 150)
        base[:, 8:] = float("nan")
        # bias holds no NaN, but its flat operand view spans base's NaN rows.
        bias = base[:, :8, :136]
        self.compile(
            lambda a, b, bias: (a @ b).reshape(5, 8, 136) + bias,
            randn(40, 72),
            randn(72, 136),
            bias,
        )
        self.assertIn("reinterpret_tensor(arg2_1, (6586, ), (1, ), 0)", self.code)
        self.assertIn("assert not arg2_1.isnan().any().item()", self.code)

    def test_horizontal(self):
        def fn(a, b, c):
            first, second = a @ b, a @ c
            return torch.relu(first - second + first * second)

        args = randn(40, 72), randn(72, 136), randn(72, 136)
        (source,) = self.compile(fn, *args)
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("acc1 = torch.addmm(acc1, tmp0, ", source)
        self.assertEqual(source.count("in0[tile_y, tile_r]"), 1)
        self.assertIn("in2[tile_r, tile_x]", source)

    def test_masked_epilogue_load(self):
        (source,) = self.compile(
            lambda a, b, first, second: a @ b + torch.cat((first, second)),
            randn(40, 72),
            randn(72, 136),
            randn(64),
            randn(72),
        )
        self.assertIn("hl.load(in2, [tile_x.index], extra_mask=", source)
        self.assertIn("hl.load(in3, [(-64) + tile_x.index], extra_mask=", source)

    @parametrize("case", ["horizontal", "vertical", "y_only"])
    def test_reduction_over_dot_loop(self, case):
        def horizontal(a, b):
            return a @ b, (a[:, :, None] * b[None]).sum(1)

        def vertical(a, b):
            # The sum reads the matmul output in a second K loop.
            c = a @ b
            return c, (c[:, :, None] * b[:, 0]).sum(-1)

        def y_only(a, b):
            return a @ b, a[:, :, None].expand(40, 72, 136).sum(1)

        # A sum over the dot's ranges reduces tile_r where the axis order puts it.
        fn, line = {
            "horizontal": (horizontal, r"acc1 = acc1 \+ torch\.sum\(tmp\d+, -2\)\n"),
            "vertical": (vertical, r"acc1 = acc1 \+ torch\.sum\(tmp\d+, -2\)\n"),
            "y_only": (y_only, r"torch\.sum\(tmp\d+, -1\)\.reshape\(\[tile_y, 1\]\)"),
        }[case]
        (source,) = self.compile(fn, randn(40, 72), randn(72, 136))
        self.assertRegex(source, line)

    @parametrize("case", ["pad", "tril", "constant", "vector", "permuted"])
    def test_operand(self, case):
        b = randn(72, 136)
        fn, a, lines = {
            # The mask needs tile_y and tile_r, and a [40, 72] view of the [36, 64]
            # input would leave its storage, so the load is flat.
            "pad": (
                lambda a, b: torch.nn.functional.pad(a, (0, 8, 0, 4)) @ b,
                randn(36, 64),
                [".reshape([tile_y, 1])", "extra_mask="],
            ),
            "tril": (
                lambda a, b: a.tril() @ b,
                randn(40, 72),
                ["torch.where("],
            ),
            "constant": (
                lambda a, b: a @ torch.ones_like(b),
                randn(40, 72),
                [".expand(tile_r, tile_x)"],
            ),
            "vector": (
                lambda a, b: a @ b[:, :1].expand(72, 136),
                randn(40, 72),
                [".reshape([tile_r, 1]).expand(tile_r, tile_x)"],
            ),
            "permuted": (
                lambda a, b: a.permute(1, 0, 2).reshape(40, 72) @ b,
                randn(8, 40, 9),
                ["//"],
            ),
        }[case]
        (source,) = self.compile(fn, a, b)
        for line in lines:
            self.assertIn(line, source)

    @parametrize("case", ["none", "dynamic_view", "native_matmul_off", "size_1"])
    def test_matmul_fallback(self, case):
        # Declined when lowering, the matmul keeps its extern kernel, and its relu
        # is a Helion kernel.
        a, b = randn(32, 64), randn(64, 48)
        fn = lambda a, b: torch.relu(a @ b)  # noqa: E731
        args = [a, b]
        if case == "dynamic_view":
            # Static sizes, but the stride and offset of h[:, -1] hold dim 1.
            fn = lambda h, b: torch.relu(h[:, -1] @ b)  # noqa: E731
            args = [randn(32, 8, 64), b]
            torch._dynamo.mark_dynamic(args[0], 1)
        elif case == "native_matmul_off":
            self.enterContext(helion_config.patch(native_matmul=False))
        elif case == "size_1":
            args = [randn(1, 64), b]
        expect_dot = case == "none"
        sources = self.compile(
            fn, *args, expected=fn(*args), tol=2e-2, expect_helion=expect_dot
        )
        self.assertEqual(len([s for s in sources if DOT_RE.search(s)]), int(expect_dot))
        self.assertEqual("extern_kernels.mm(" in self.code, not expect_dot)

    def test_matmul_row_reduction(self):
        # A sum over the rows of a @ b with N <= 128: the dot and the sum are
        # Helion kernels of their own, not SIMDScheduling's fused row epilogue.
        # FP32, since in BF16 the sum adds a @ b rounded to BF16, unlike FP32 eager.
        a, b = randn(32, 64, dtype=torch.float32), randn(64, 48, dtype=torch.float32)
        sources = self.compile(lambda a, b: (a @ b).sum(-1), a, b, tol=1e-3)
        self.assertEqual([bool(DOT_RE.search(s)) for s in sources], [True, False])

    def test_matmul_unsupported(self):
        # Lowered to a dot, which a kernel that Helion cannot emit holds: a[i]
        # needs a bounds check, which Helion kernels lack (test_gathered_operand
        # has none).
        b = randn(64, 48)
        i = torch.randint(0, 80, (32,), device=DEVICE)
        self.assertUnsupported(
            "check_bounds", lambda a, i, b: a[i] @ b, randn(80, 64), i, b
        )
        self.assertUnsupported(
            "cpu_scalar_input",
            lambda a, b, c: a @ b + c,
            randn(32, 64),
            b,
            torch.tensor(2.0),
        )

    def test_gathered_operand(self):
        # a[i] @ b with assert_indirect_indexing=False (with it, the check of i
        # fails the compile, test_matmul_unsupported): the dot reads a's rows at
        # i[tile_y], [tile_y] against [tile_y, tile_r].
        a, b = randn(80, 64), randn(64, 48)
        i = torch.randint(0, 80, (32,), device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            (source,) = self.compile(lambda a, i, b: a[i] @ b, a, i, b, tol=2e-2)
        self.assertIn("torch.addmm(", source)

    def test_cudagraph_tree(self):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        args = [randn(32, 64), randn(64, 48)]
        counters.clear()
        with config.patch({"triton.cudagraphs": True}):
            compiled = torch.compile(fn, fullgraph=True)
            torch.compiler.cudagraph_mark_step_begin()
            result, (code,) = run_and_get_code(compiled, *args)
            self.assertEqual(result, fn(*args))
            for _ in range(2):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
        self.assertEqual(len(SOURCE_RE.findall(code)), 2)


# The CuTe lane, where CuTe DSL is installed, runs this.
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestInductorCute(InductorBackendTestCase):
    @needs_native_matmul
    def test_bmm_epilogue(self):
        self.check_bmm_epilogue("cute")

    def test_unsupported_op(self):
        # Helion's CuTe backend compiles no lgamma.
        x = torch.rand(64, 100, device=DEVICE) + 0.5
        self.assertUnsupported("lgamma", torch.lgamma, x, backend="cute")
        (source,) = self.compile(torch.exp, x, tol=1e-4, backend="cute")
        self.assertIn('backend="cute"', source)

    def test_arg_reduction(self):
        # arg_reduce()'s combine emits isnan, logical_and/or/not and an int64
        # where, which backend_ops() does not see in the node's body.
        x = torch.randn(64, 100, device=DEVICE)
        x[3, 10] = float("nan")
        fn = lambda x: (torch.argmax(x, -1), torch.any(x > 2.0, -1))  # noqa: E731
        (source,) = self.compile(fn, x, backend="cute")
        self.assertIn("torch.any(", source)

    def test_welford(self):
        # welford() emits eq, where, truediv, isinf, isnan and logical_or, and
        # the block's weight sums hl.full([], 1.0).expand(tile_r).
        x = torch.randn(64, 1000, device=DEVICE)
        (source,) = self.compile(lambda x: torch.var_mean(x, -1), x, backend="cute")
        self.assertIn(".expand(tile_r)", source)

    def test_rand(self):
        # Stock Inductor's numbers, bit for bit: CuTe's default Philox4 stream
        # draws others.
        x = torch.randn(64, 1000, device=DEVICE)
        expected = self.stock_inductor(torch.rand_like, x)
        (source,) = self.compile(
            torch.rand_like, x, expected=expected, tol=0, backend="cute"
        )
        self.assertIn('cute_rng_stream="word0"', source)

    def test_gather(self):
        x = torch.randn(64, 100, device=DEVICE)
        i = torch.randint(0, 100, (64, 10), device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            (source,) = self.compile(
                lambda x, i: x.gather(1, i) * 2, x, i, backend="cute"
            )
        self.assertIn("in1[tmp1 + ", source)


@skipIfNotCUDA()
@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestInductorBackend(InductorBackendTestCase):
    def test_needs_torch_2_16(self):
        # importlib.reload() stops at the check, so the module keeps its state but
        # the requires_torch_version it rebinds before the check.
        real = helion.inductor.requires_torch_version
        self.addCleanup(setattr, helion.inductor, "requires_torch_version", real)
        with (
            patch("helion._compat.requires_torch_version", return_value=False),
            self.assertRaisesRegex(ImportError, "needs torch 2.16 or later"),
        ):
            importlib.reload(helion.inductor)

    def test_cuda_backend(self):
        # import helion.inductor makes "helion" a cuda_backend, and keeps the others.
        scheduling = common.get_scheduling_for_device("cuda")
        self.assertIsInstance(scheduling(None), helion.inductor.HelionScheduling)
        for backend in ("triton", "halide"):
            with config.patch(cuda_backend=backend):
                self.assertNotIsInstance(
                    scheduling(None), helion.inductor.HelionScheduling
                )
        self.assertIn("helion", dict(registered_inductor_choices()))

    @parametrize("case", ["distinct", "repeated", "rows", "one_element"])
    def test_atomic_store(self, case):
        # index_put(accumulate=True) at an arange-based index is an atomic_add
        # store with a direct index, where repeated indices add up.
        def fn(x):
            n = torch.arange(x.size(0), device=x.device)
            index = {
                "distinct": n * 2,
                "repeated": n // 2,
                "rows": n // 2,
                "one_element": n * 0,
            }[case]
            out = torch.zeros(2 * x.size(0), *x.shape[1:], device=x.device)
            return out.index_put((index,), x, True)

        x = torch.randn(64, *([8] if case == "rows" else []), device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            sources = self.compile(fn, x, tol=1e-4)
        self.assertEqual(len([s for s in sources if "hl.atomic_add(" in s]), 1)
        if case == "one_element":
            # Every x[i] adds at out[0], through a stride-0 view.
            self.assertRegex(
                self.code, r"reinterpret_tensor\(buf\d+, \(64, \), \(0, \), 0\)"
            )

    def test_keeps_cuda_custom_pass(self):
        custom_pass = object()
        with (
            patch.dict(common.device_codegens),
            patch.dict(common.custom_backend_passes, {"cuda": custom_pass}),
        ):
            helion.inductor._register()
            self.assertIs(common.custom_backend_passes["cuda"], custom_pass)

    def test_options(self):
        # The settings are torch.compile options with a "helion." prefix.
        x = torch.randn(64, 100, device=DEVICE)
        options = {"cuda_backend": "helion", "helion.fast_math": True}
        with config.patch(cuda_backend="triton"):
            _, (code,) = run_and_get_code(torch.compile(torch.exp, options=options), x)
            self.assertIn("fast_math=True", code)
            with self.assertRaisesRegex(RuntimeError, "helion.fast_mathh"):
                torch.compile(torch.exp, options={"helion.fast_mathh": True})(x)

    def test_helion_kernel_in_graph(self):
        # A @helion.kernel called in the compiled function is Helion's own
        # template, here with relu and * 2 fused into it (torch_compile_fusion).
        x, y = torch.randn(64, 128, device=DEVICE), torch.randn(64, 128, device=DEVICE)
        fn = lambda x, y: add_kernel(x, y).relu() * 2  # noqa: E731
        self.compile(fn, x, y, expected=fn(x, y), expect_helion=False)
        self.assertRegex(self.code, r"def _helion_add_\w*_fused_mul_relu_0\(")

    def test_helion_kernel_numerics(self):
        # Helion lowers a @helion.kernel through Inductor as for cuda_backend
        # "triton", with the fp32 constant of x * 0.1 for a bf16 x.
        code, _ = code_and_output(scale_kernel, (randn(64, 128),))
        self.assertIn("tl.full([], 0.1, tl.float32)", code)

    def test_softmax(self):
        # The sum's value derives from the max: exp(x - max), whose dtype and
        # shape HelionCSEProxy propagates.
        (source,) = self.compile(
            lambda x: torch.softmax(x, -1), torch.randn(64, 1000, device=DEVICE)
        )
        self.assertIn("acc1 = acc1 + torch.sum(", source)

    @parametrize("op", ["cumsum", "sort"])
    def test_aten_fallback(self, op):
        # No BackendFeature.SCAN or SORT: Inductor lowers them to aten kernels.
        fn = {
            "cumsum": lambda x: torch.cumsum(x, -1) * 2,
            "sort": lambda x: torch.sort(x, -1).values * 2,
        }[op]
        (source,) = self.compile(fn, torch.randn(64, 100, device=DEVICE))
        self.assertRegex(self.code, rf"aten\.{op}\.\w+\(")

    def test_reduction_source(self):
        (source,) = self.compile(
            lambda x, b: (x * 2).sum(-1) + b,
            torch.randn(64, 100, device=DEVICE),
            torch.randn(64, device=DEVICE),
            tol=1e-4,
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(64):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(100):
            tmp0 = in0[tile_x, tile_r]
            tmp1 = hl.full([], 2.0, dtype=torch.float32)
            tmp2 = tmp0 * tmp1
            acc = acc + torch.sum(tmp2, -1)
        tmp3 = in1[tile_x]
        tmp4 = acc + tmp3
        out0[tile_x] = tmp4
""",
        )

    @parametrize("op", ["sum", "prod"])
    def test_integer_reduction(self, op):
        # -1 and 1, so that the product is rarely 0 and does not overflow.
        x = torch.randint(0, 2, (64, 100), device=DEVICE, dtype=torch.int32) * 2 - 1
        fn = lambda x: getattr(x, op)(-1, dtype=torch.int32)  # noqa: E731
        (source,) = self.compile(fn, x)
        self.assertRegex(source, rf"torch\.{op}\(tmp\d+, -1, dtype=torch\.int32\)")

    @config.patch({"fx_graph_cache": True, "force_disable_caches": False})
    def test_cache_key_has_source(self):
        # helion.inductor registers helion_config with Inductor, whose FX graph
        # cache key includes its private _source_hash: another helion.inductor
        # misses.
        x = torch.randn(64, 100, device=DEVICE)
        counters.clear()
        with fresh_cache():
            for source_hash in ("a", "a", "b"):
                torch._dynamo.reset()
                with helion_config.patch(_source_hash=source_hash):
                    torch.compile(lambda x: x.sum(-1))(x)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @parametrize(
        "case",
        ["argmax", "min_dim", "max_dim", "nan", "ties", "int", "neg_inf", "flat"],
    )
    def test_arg_reduction(self, case):
        # min_dim: torch.min(x, -1) with a NaN, an amin and an argmin; nan:
        # torch.argmax picks the first NaN; ties: argmin's first index; int: no
        # NaN checks; neg_inf: a row of -inf, where no block is larger than the
        # accumulator's -inf and the first index wins the tie; flat: a reduction
        # over both dims of a transposed x, Unsupported (unsupported()).
        x = torch.randn(64, 300, device=DEVICE)
        fn = lambda x: torch.argmax(x, -1)  # noqa: E731
        if case == "min_dim":
            x[4, 100] = float("nan")
            fn = lambda x: torch.min(x, -1)  # noqa: E731
        elif case == "max_dim":
            fn = lambda x: torch.max(x, -1)  # noqa: E731
        elif case == "nan":
            x[3, 10] = x[3, 250] = x[5, 200] = float("nan")
        elif case == "ties":
            x = torch.randint(0, 3, (64, 300), device=DEVICE).float()
            fn = lambda x: torch.argmin(x, -1)  # noqa: E731
        elif case == "int":
            x = torch.randint(-50, 50, (64, 300), device=DEVICE, dtype=torch.int32)
        elif case == "neg_inf":
            x[2] = float("-inf")
        elif case == "flat":
            x = torch.randn(300, 64, device=DEVICE).t()
            self.assertUnsupported("arg_reduction_over_dims", torch.argmax, x)
            return
        (source,) = self.compile(fn, x)
        self.assertIn("torch.amin(", source)

    def test_welford(self):
        (source,) = self.compile(
            lambda x: torch.var_mean(x, -1), torch.randn(64, 100, device=DEVICE)
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_x in hl.tile(64):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(100):
            tmp0 = in0[tile_x, tile_r]
            tmp1 = hl.full([], 0.0, dtype=torch.float32)
            tmp2 = hl.full([], 1.0, dtype=torch.float32)
            tmp3 = tmp2.expand(tile_r)
            tmp4 = torch.sum(tmp3, -1)
            tmp5 = tmp2 / tmp4
            tmp6 = tmp0 * tmp5
            tmp7 = torch.sum(tmp6, -1)
            tmp8 = torch.isinf(tmp7)
            tmp9 = torch.isnan(tmp7)
            tmp10 = torch.logical_or(tmp8, tmp9)
            tmp11 = torch.where(tmp10, tmp1, tmp7)
            tmp12 = tmp11.reshape([tile_x, 1])
            tmp13 = tmp0 - tmp12
            tmp14 = tmp13 * tmp5
            tmp15 = torch.sum(tmp14, -1)
            tmp16 = tmp11 + tmp15
            tmp17 = tmp16.reshape([tile_x, 1])
            tmp18 = tmp0 - tmp17
            tmp19 = tmp18 * tmp5
            tmp20 = torch.sum(tmp19, -1)
            tmp21 = tmp20.reshape([tile_x, 1])
            tmp22 = tmp18 - tmp21
            tmp23 = tmp22 * tmp22
            tmp24 = torch.sum(tmp23, -1)
            tmp25 = acc2 + tmp4
            tmp26 = acc == tmp16
            tmp27 = tmp16 - acc
            tmp28 = torch.where(tmp26, tmp1, tmp27)
            tmp29 = tmp25 == tmp1
            tmp30 = tmp4 / tmp25
            tmp31 = torch.where(tmp29, tmp1, tmp30)
            tmp32 = tmp28 * tmp31
            tmp33 = acc + tmp32
            tmp34 = acc1 + tmp24
            tmp35 = acc2 * tmp31
            tmp36 = tmp28 * tmp35
            tmp37 = tmp36 * tmp28
            tmp38 = tmp34 + tmp37
            acc = tmp33
            acc1 = tmp38
            acc2 = tmp25
        out0[tile_x] = acc
        tmp39 = hl.full([], 99.0, dtype=torch.float32)
        tmp40 = acc1 / tmp39
        out1[tile_x] = tmp40
""",
        )

    @parametrize("case", ["split", "rows"])
    def test_welford_cases(self, case):
        # split: a kernel that reduces chunks of each row to (mean, m2, weight),
        # then a welford_combine kernel over them. R = 65537 (few rows and a
        # long R, which Inductor splits on any GPU) is not a multiple of the
        # chunk, so the first kernel is a welford_combine over (x, 0, 1), with
        # (0, 0, 0) past the end of R. rows, one block each: rows 0-7 with one
        # value of 2000, as in some channels of LLM activations, whose mean is
        # as close to eager's as Triton's; rows 8-15 of 1e37, whose sum
        # overflows; rows 16-23 of 100000000 and 100000008 in turn, whose mean
        # rounds.
        fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
        if case == "split":
            x = torch.randn(8, 65537, device=DEVICE) + 100
            self.assertEqual(len(self.compile(fn, x)), 2)
            # Halves of 1e37 and -1e37: partial means of 1e37 with weights of
            # thousands, whose product overflows. The mean is finite and the
            # variance overflows, as in eager.
            x[:, :32768], x[:, 32768:] = 1e37, -1e37
            var, mean = torch.compile(fn, fullgraph=True)(x)
            self.assertTrue(torch.isinf(var).all() and torch.isfinite(mean).all())
            return
        x = torch.randn(24, 4096, device=DEVICE)
        x[:8, 100], x[8:16], x[16:] = 2000.0, 1e37, 100000000.0
        x[16:, ::2] = 100000008.0
        self.assertEqual(len(self.compile(fn, x)), 1)

    def test_welford_empty_chunks(self):
        # A reduction_split_factor() of 1024 cuts R = 9000 into chunks of 9, so
        # the last 24 chunks hold nothing past the end of R.
        class Choices(InductorChoices):
            def reduction_split_factor(
                self,
                device: torch.device,
                reduction_numel_hint: int,
                numel_hint: int,
                inner_reduction: bool,
            ) -> int:
                return 1024 if reduction_numel_hint == 9000 else 1

        x = torch.randn(8, 9000, device=DEVICE)
        with V.set_choices_handler(Choices()):
            sources = self.compile(lambda x: torch.var_mean(x, -1), x)
        self.assertEqual(len(sources), 2)

    def test_welford_layer_norm(self):
        # bf16 with its affine epilogue, in one kernel.
        x, w, b = (
            torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
            for shape in ((64, 1000), (1000,), (1000,))
        )
        expected = torch.nn.functional.layer_norm(
            x.float(), (1000,), w.float(), b.float()
        ).bfloat16()
        fn = lambda x: torch.nn.functional.layer_norm(x, (1000,), w, b)  # noqa: E731
        self.assertEqual(len(self.compile(fn, x, expected=expected, tol=2e-2)), 1)

    @parametrize("case", ["rows", "split", "split_empty"])
    def test_welford_extremes(self, case):
        # Blocks of tile_r pinned to 64, or 1 for split_empty (by default one
        # block covers a row). rows, R = 963, whose last block has 3 lanes:
        # rows 1 and 2, of inf, get the mean inf, as from Triton's persistent
        # reductions (eager gives nan), since the delta of two equal means is
        # 0, where inf - inf is nan. Row 3, of 2e19, gets the variance 0, as on
        # its first block acc_weight is 0 and delta 2e19, whose square is inf in
        # float32. Row 4, of 100553424 (c), gets the variance 0, as the last
        # block's mean is c, where (c + c + c) / 3 is 100553432. Row 5, of 1e38
        # and 0 in turn, gets the mean 5e37: its deviations of 5e37 are divided
        # by the weight before they are summed. split: test_welford_cases'
        # masked split, whose first kernel has partials of weight 0 and mean 0
        # past the end of R. Rows of -134217720, whose blocks of unequal weight
        # give their mean exactly only with the correction, and of 2e19, whose
        # squared deviation from that 0 is inf, get the variance 0.
        # split_empty: the same split, whose last block of the last chunk holds
        # only a partial of weight 0, whose mean would be 0 / 0.
        rows = case == "rows"
        if rows:
            x = torch.randn(6, 963, device=DEVICE)
            x[1], x[2, 480:] = float("inf"), float("inf")
            x[3], x[4], x[5] = 2e19, 100553424.0, 0.0
            x[5, ::2] = 1e38
        elif case == "split":
            x = torch.full((8, 65537), -134217720.0, device=DEVICE)
            x[4:] = 2e19
        else:
            x = torch.randn(8, 65537, device=DEVICE)
        var, mean = torch.var_mean(x, -1)
        if rows:
            mean[1:3] = float("inf")
        pinned = helion.Config(block_sizes=[1, 1 if case == "split_empty" else 64])
        with patch("helion.kernel", functools.partial(helion.kernel, config=pinned)):
            fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
            sources = self.compile(fn, x, expected=(var, mean))
        self.assertEqual(len(sources), 1 if rows else 2)

    def test_gather(self):
        x = torch.randn(64, 100, device=DEVICE)
        i = torch.randint(0, 100, (64, 10), device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            (source,) = self.compile(lambda x, i: x.gather(1, i) * 2, x, i)
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(640):
        tmp0 = in0[tile_x]
        tmp1 = tmp0.to(torch.int64)
        tmp2 = in1[tmp1 + 100*(tile_x.index // 10)]
        tmp3 = hl.full([], 2.0, dtype=torch.float32)
        tmp4 = tmp2 * tmp3
        out0[tile_x] = tmp4
""",
        )

    @parametrize(
        "case",
        [
            "embedding",
            "embedding_norm",
            "maxpool_offsets",
            "index_select",
            "negative",
            "arange",
            "cross_entropy",
            "argmax",
            "masked_index",
            "masked_constant",
            "index_put",
            "scatter_add",
        ],
    )
    def test_gather_cases(self, case):
        # One case per lowering of an indirect index (12 > 8). embedding_norm:
        # rows read at i[tile_x] in a reduction over their columns, at the index
        # 64 * i + r of [tile_x, tile_r]. maxpool_offsets: max_pool2d's backward
        # over a width of 300, which scatters at 298 * (o // 2) + ... of its int8
        # offsets o. negative: x[i] with i < 0 reads x[i + 100], at an index that
        # is only the value read. arange: an index computed from iota,
        # which the body holds as a direct index and a check_bounds that passes,
        # so it is Helion with assert_indirect_indexing. cross_entropy: the
        # gather of the target's logit in a reduction kernel. argmax: an index
        # that is a fused reduction's accumulator. masked_index: a load at one
        # index for all lanes, out of range, under a mask that turns them off;
        # masked_constant: the same at a constant index, a direct load.
        # index_put: a store at an index read from a tensor (no repeats), and
        # scatter_add an atomic add.
        x, w = torch.randn(64, 100, device=DEVICE), torch.randn(1000, 64, device=DEVICE)
        ids = torch.randint(0, 1000, (32,), device=DEVICE)
        fn, args = {
            "embedding": (
                lambda i, w: torch.nn.functional.embedding(i, w) * 3,
                (ids, w),
            ),
            "embedding_norm": (
                lambda i, w: torch.nn.functional.embedding(i, w).softmax(-1),
                (ids, w),
            ),
            "maxpool_offsets": (
                lambda o, g: torch.zeros(2, 1200, device=g.device).scatter_add(
                    1,
                    torch.ops.prims._low_memory_max_pool_offsets_to_indices(
                        o, [2, 2], [4, 300], [2, 2], [0, 0], [1, 1]
                    ).view(2, 300),
                    g.view(2, 300),
                ),
                (
                    torch.randint(0, 4, (1, 2, 2, 150), device=DEVICE).to(torch.int8),
                    torch.randn(1, 2, 2, 150, device=DEVICE),
                ),
            ),
            "index_select": (lambda w, i: w.index_select(0, i) + 1, (w, ids)),
            "negative": (
                lambda x, i: x[i] * 2,
                (x[0], torch.randint(-100, 100, (64,), device=DEVICE)),
            ),
            "arange": (
                lambda x: x.gather(
                    1, torch.arange(10, device=x.device).expand(64, 10) * 7
                ),
                (x,),
            ),
            "cross_entropy": (
                torch.nn.functional.cross_entropy,
                (
                    torch.randn(128, 1000, device=DEVICE),
                    torch.randint(0, 1000, (128,), device=DEVICE),
                ),
            ),
            "argmax": (
                lambda x, y: y.gather(1, x.argmax(1, keepdim=True)),
                (x, torch.randn(64, 100, device=DEVICE)),
            ),
            "masked_index": (
                lambda w, m, i: torch.ops.aten._unsafe_masked_index(w, m, [i], 0.0),
                (
                    w.flatten(),
                    torch.zeros(64, dtype=torch.bool, device=DEVICE),
                    torch.tensor([2**40], device=DEVICE).expand(64),
                ),
            ),
            "masked_constant": (
                lambda w, m: torch.ops.aten._unsafe_masked_index(
                    w, m, [torch.full((64,), 2**40, device=w.device)], 0.0
                ),
                (w.flatten(), torch.zeros(64, dtype=torch.bool, device=DEVICE)),
            ),
            "index_put": (
                lambda i, v: torch.zeros(1000, 64, device=v.device).index_put((i,), v),
                (
                    torch.randperm(1000, device=DEVICE)[:32],
                    torch.randn(32, 64, device=DEVICE),
                ),
            ),
            "scatter_add": (
                lambda i, s: torch.zeros(64, 100, device=s.device).scatter_add(1, i, s),
                (
                    torch.randint(0, 100, (64, 10), device=DEVICE),
                    torch.randn(64, 10, device=DEVICE),
                ),
            ),
        }[case]
        with config.patch(assert_indirect_indexing=case == "arange"):
            self.compile(fn, *args, tol=1e-4)
        self.assertNotIn("@triton.jit", self.code)

    @skipIfLowVRAM("x takes 2 GiB", required_bytes=3 * 2**30)
    def test_gather_wide_index(self):
        # x[i] with an int32 i into x of 2**31 + 8 elements (2 GB of uint8): -1
        # reads x at i + 2147483656, past int32.
        x = torch.empty(2**31 + 8, dtype=torch.uint8, device=DEVICE)
        x[-1] = 7
        i = torch.tensor([-1, 5, 2**31 - 1], dtype=torch.int32, device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            self.compile(lambda x, i: x[i] + 1, x, i)

    @parametrize("case", ["check_bounds", "dynamic_index", "where_index"])
    def test_gather_unsupported(self, case):
        # check_bounds: Inductor checks an index of unknown range, Triton with
        # tl.device_assert; Helion kernels have none. dynamic_index: x[i] into a
        # dynamic dim, which a negative i wraps by. where_index: x[arange - 1], at
        # the direct index Where(p0 - 1 < 0, p0 + 9, p0 - 1).
        x = torch.randn(64, 100, device=DEVICE)
        if case == "check_bounds":
            fn = lambda x, i: x.gather(1, i) * 2  # noqa: E731
            args = (x, torch.randint(0, 100, (64, 10), device=DEVICE))
        elif case == "where_index":
            fn = lambda x: x[torch.arange(10, device=x.device) - 1] + 1  # noqa: E731
            args = (x,)
        else:
            torch._dynamo.mark_dynamic(x, 0)
            fn = lambda x, i: x[i] * 2  # noqa: E731
            args = (x, torch.randint(-64, 64, (32,), device=DEVICE))
        with config.patch(assert_indirect_indexing=case == "check_bounds"):
            self.assertUnsupported(case, fn, *args)

    def test_any(self):
        x = torch.randn(64, 100, device=DEVICE)
        (source,) = self.compile(lambda x: torch.any(x > 2.0, -1), x)
        self.assertIn("acc = acc | torch.any(", source)

    def test_pointwise(self):
        (source,) = self.compile(
            lambda x: torch.relu(x) * 2 + 1, torch.randn(64, 100, device=DEVICE)
        )
        self.assertIn("for tile_x in hl.tile(6400):", source)

    @parametrize("case", ["float_to_uint8", "int_to_fp8", "bool_to_fp8"])
    def test_cast(self, case):
        x = torch.randn(64, 100, device=DEVICE) * 100
        fn, line = {
            # eager wraps a negative value: (-5.3).to(torch.uint8) is 251.
            "float_to_uint8": (
                lambda x: (x * 2 - 1).to(torch.uint8),
                ".to(torch.int16).to(torch.uint8)",
            ),
            "int_to_fp8": (
                lambda x: (x.int() + 1).to(torch.float8_e4m3fn).float(),
                ".to(torch.float32).to(torch.float8_e4m3fn)",
            ),
            "bool_to_fp8": (
                lambda x: (x > 0).to(torch.float8_e5m2).float(),
                ".to(torch.float32).to(torch.float8_e5m2)",
            ),
        }[case]
        if case == "int_to_fp8" and torch.cuda.get_device_capability() < (8, 9):
            # "Conversion from/to f8e4m3nv is only supported on compute capability
            # >= 89"
            self.skipTest("Triton converts to float8_e4m3fn on SM89 and later")
        (source,) = self.compile(fn, x)
        self.assertIn(line, source)

    def test_inplace_reuse_chain(self):
        def fn(x, w):
            y = torch.nn.functional.conv2d(x, w, padding=1)
            for i in range(32):
                y = y * 0.999 + 0.001 * (i + 1)
            return y + y.cos()

        x = torch.randn(1, 1, 64, 128, device=DEVICE)
        w = torch.randn(1, 1, 3, 3, device=DEVICE)
        (source,) = self.compile(fn, x, w, tol=1e-4)
        # The conv output buf0 becomes buf1, which becomes buf2, so buf0 is passed as buf2.
        self.assertIn("buf2 = buf1; del buf1  # reuse", self.code)
        self.assertRegex(self.code, r"helion_\w+\(reinterpret_tensor\(buf2, ")

    def test_rand(self):
        # rand_like(x) draws stock Inductor's tl.rand numbers, bit for bit.
        x = torch.randn(64, 1000, device=DEVICE)
        expected = self.stock_inductor(torch.rand_like, x)
        (source,) = self.compile(torch.rand_like, x, expected=expected, tol=0)
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(64000):
        tmp0 = in0[()]
        tmp1 = (tile_x.index).to(torch.int32)
        tmp2 = hl.rand([], seed=tmp0, offsets=tmp1.to(torch.int64) & 0xFFFFFFFF)
        out0[tile_x] = tmp2
""",
        )

    @parametrize("case", ["pointwise", "reduction", "two_seeds"])
    def test_dropout(self, case):
        # Stock Inductor's masks, bit for bit. reduction: a dropout in a
        # softmax's kernel; two_seeds: two dropouts, which read seeds[0] and
        # seeds[1] through 0-d views of the seed buffer.
        x = torch.randn(64, 1000, device=DEVICE)
        drop = torch.nn.functional.dropout
        fn = {
            "pointwise": lambda x: drop(x, 0.5) == 0,
            "reduction": lambda x: drop(torch.softmax(x, -1), 0.1) == 0,
            "two_seeds": lambda x: (drop(x, 0.3) == 0) & (drop(x * 2, 0.6) == 0),
        }[case]
        (source,) = self.compile(fn, x, expected=self.stock_inductor(fn, x))
        self.assertEqual(source.count("hl.rand("), 2 if case == "two_seeds" else 1)

    @pytest.mark.slow
    @skipIfLowVRAM("two masks of 2 GiB", required_bytes=8 * 2**30)
    def test_rand_wide_offset(self):
        # 2**31 + 16 elements: Inductor's int32 offset wraps for the last 16,
        # which stock Inductor draws at offset.to(tl.uint32).
        fn = lambda x: torch.rand(2**31 + 16, device=x.device) > 0.5  # noqa: E731
        x = torch.zeros(1, device=DEVICE)
        self.compile(fn, x, expected=self.stock_inductor(fn, x))

    # One case per name that unsupported() and HelionScheduling report (9 > 8:
    # each is a separate check); test_gather_unsupported has the index ones.
    @parametrize(
        "case",
        [
            "xor_sum",
            "randn",
            "dynamic",
            "strict_reduction",
            "batch_invariant",
            "deterministic",
            "pytorch_libdevice",
            "cpp_wrapper",
            "combo_kernels",
        ],
    )
    def test_unsupported(self, case):
        # What the kernel uses that HelionKernel does not support, by name.
        x = torch.randn(64, 100, device=DEVICE)
        fn = lambda x: x.sum(-1)  # noqa: E731
        args = [x]
        if case == "xor_sum":
            # The one reduction type of Inductor's that Helion kernels lack.
            fn = lambda x: torch.ops.prims.xor_sum(x, [1])  # noqa: E731
            args = [torch.randint(0, 100, (64, 100), device=DEVICE, dtype=torch.int32)]
        elif case == "randn":
            fn = lambda x: x + torch.randn_like(x)  # noqa: E731
        elif case == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        elif case == "combo_kernels":
            fn = lambda x, y: (x * 2, y + 1)  # noqa: E731
            args = [x, torch.randn(32, 50, device=DEVICE)]
        names, patches = {
            "xor_sum": ("xor_sum", {}),
            "randn": ("randn", {}),
            "dynamic": ("dynamic_shape", {}),
            "strict_reduction": ("strict_reduction", {"numerics": "strict_reduction"}),
            "batch_invariant": ("batch_invariant", {"batch_invariant": True}),
            "deterministic": ("deterministic", {"deterministic": True}),
            "pytorch_libdevice": (
                "eager_numerics.use_pytorch_libdevice",
                {"eager_numerics.use_pytorch_libdevice": True},
            ),
            "cpp_wrapper": ("cpp_wrapper", {"cpp_wrapper": True}),
            "combo_kernels": ("combo_kernels", {"combo_kernels": True}),
        }[case]
        with config.patch(patches):
            self.assertUnsupported(names, fn, *args)

    def test_broadcast_reduction(self):
        # The max over a broadcast dim of z, all -0.0, is -0.0, as in eager.
        z = torch.full((64, 1), -0.0, device=DEVICE)
        fn = lambda z: torch.signbit(torch.amax(z.expand(64, 300), -1))  # noqa: E731
        (source,) = self.compile(fn, z)
        self.assertIn(".expand(tile_x, tile_r)", source)

    @config.patch({"triton.codegen_upcast_to_fp32": False})
    def test_reduction_low_precision(self):
        # A bf16 value sums into an fp32 accumulator, as in Triton kernels.
        (source,) = self.compile(lambda x: x.sum(-1), randn(64, 1000))
        self.assertIn("acc = hl.zeros([tile_x], dtype=torch.float32)", source)
        self.assertIn("= tmp0.to(torch.float32)", source)

    def test_many_size_1_dims(self):
        # out0 has 25 dims of size 1, and tile_x has 1 element.
        x = torch.randn(*[1] * 24, 128, device=DEVICE)
        self.compile(lambda x: x.sum(-1, keepdim=True), x, tol=1e-4)

    def test_fast_math(self):
        self.enterContext(helion_config.patch(fast_math=True))
        fn = torch.nn.functional.gelu
        (source,) = self.compile(fn, torch.randn(64, 100, device=DEVICE), tol=1e-4)
        self.assertIn("fast_math=True", source)


if __name__ == "__main__":
    unittest.main()
