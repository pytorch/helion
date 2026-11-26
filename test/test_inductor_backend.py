from __future__ import annotations

import functools
import importlib
import inspect
import re
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor import metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen import common
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
import torch.nn.functional as F
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import subtest
import torch.utils._pytree as pytree

from helion._compat import requires_torch_version
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfFbTriton
from helion._testing import skipIfLowVRAM
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager

if not requires_torch_version("2.16"):
    # helion.inductor uses the Inductor of torch 2.16, e.g. config.strict_reduction.
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

# After the check: torch 2.13 and 2.14 do not have it.
from torch._inductor.choices import registered_inductor_choices
from torch._inductor.runtime.hints import DeviceProperties

import helion
from helion._compiler.cute.inductor_plan import transposed_product_wins
from helion._compiler.cute.tcgen05.search_plan import (
    tcgen05_transposes_epilogue_bound_product,
)
from helion._compiler.cute.tcgen05.search_plan import (
    tcgen05_transposes_misaligned_b_rows,
)
from helion._compiler.cute.tcgen05.search_plan import tcgen05_transposes_skinny_product
import helion.inductor
from helion.inductor import config as helion_config
from helion.inductor import kernel_sequence
from helion.inductor.codegen.kernel import exact_cast
import helion.language as hl

SOURCE_RE = re.compile(r'helion_\w+_source = """\\\n(.*?)"""', re.DOTALL)
DOT_RE = re.compile(r"torch\.(?:addmm|baddbmm|mm|bmm)\(")


# Inductor lowers matmuls natively for a backend through
# InductorChoices.use_native_matmul (pytorch/pytorch#204025).
needs_native_matmul = unittest.skipUnless(
    "use_native_matmul" in vars(InductorChoices),
    "torch without InductorChoices.use_native_matmul",
)


def transposes(m: int, n: int, k: int, *, b_n_major: bool = False) -> bool:
    """Whether helion.inductor's pair-tile rule transposes a 16-bit m x n x k RC (RR)
    product on this GPU."""
    return transposed_product_wins(m, n, k, torch.device(DEVICE), b_n_major=b_n_major)


def skinny_transposes(m: int, n: int, k: int, *, b_n_major: bool) -> bool:
    """Whether tcgen05_transposes_skinny_product() transposes a skinny product on this
    GPU."""
    props = DeviceProperties.create(torch.device(DEVICE))
    return props.major == 10 and tcgen05_transposes_skinny_product(
        m,
        n,
        k,
        torch.bfloat16,
        a_strides=(k, 1),
        b_strides=(n, 1) if b_n_major else (1, k),
        num_sms=props.multi_processor_count,
    )


def misaligned_transposes(m: int, n: int, k: int) -> bool:
    """Whether tcgen05_transposes_misaligned_b_rows() transposes an RR product on this
    GPU."""
    props = DeviceProperties.create(torch.device(DEVICE))
    return props.major == 10 and tcgen05_transposes_misaligned_b_rows(
        m,
        n,
        k,
        torch.bfloat16,
        a_strides=(k, 1),
        b_strides=(n, 1),
        num_sms=props.multi_processor_count,
    )


def epilogue_bound_transposes(m: int, n: int, k: int) -> bool:
    """Whether tcgen05_transposes_epilogue_bound_product() transposes an RC product on
    this GPU."""
    props = DeviceProperties.create(torch.device(DEVICE))
    return props.major == 10 and tcgen05_transposes_epilogue_bound_product(
        m,
        n,
        k,
        torch.bfloat16,
        a_strides=(k, 1),
        b_strides=(1, k),
        num_sms=props.multi_processor_count,
    )


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device=DEVICE, dtype=dtype)


# For the tests of a loop over tile_r, its accumulators and their combine across
# blocks: by default a reduction is one block, a persistent reduction, at any
# length, and those that loop as Triton's do (choices.LOOPED_REDUCTIONS) up to
# Triton's 1024 elements of an inner reduction or 64 of another
# (HelionChoices.should_use_persistent_reduction()).
looped = config.patch({"triton.persistent_reductions": False})


def conv_inplace(x, w):
    """A conv's output that its kernel adds 1 to and stores again, an in-place
    buffer (buf1 takes over the conv's buf0), and its rdims' mean square."""
    y = torch.nn.functional.conv2d(x, w, padding=1) + 1.0
    return y, y.pow(2).mean(-1)


def pin_blocks(r_block: int):
    """A patch of helion.kernel that pins each kernel's blocks: 1 for its outer
    tiles and r_block for the tile of its inner loop (tile_r, or a mix-order
    kernel's tile_x). A split's first kernel tiles its x and chunks of R,
        for tile_y, tile_x in hl.tile([8, 64]):
    its second kernel x alone."""
    kernel = helion.kernel

    def pinned(fn=None, **kwargs):
        if fn is None:  # @helion.kernel(backend=..., ...)
            return functools.partial(pinned, **kwargs)
        source = inspect.getsource(fn)
        outer = re.search(r"for (.*) in hl\.tile\(", source).group(1)
        blocks = [1] * (outer.count(",") + 1) + [r_block]
        return kernel(fn, config=helion.Config(block_sizes=blocks), **kwargs)

    return patch("helion.kernel", pinned)


# The load of x in test_masked_view: through its own view, tiled as the output.
MASKED_VIEW_LOAD = r"hl\.load\(in0, \[tile_y, tile_x\], extra_mask=tmp\d+\)"


class PersistentChoices(InductorChoices):
    """Choices that make every reduction persistent."""

    @staticmethod
    def should_use_persistent_reduction(features, cooperative_reduction):
        return True


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

    def check_mix_order_reduction(self, backend):
        """RMSNorm's backward, given rstd: its reduction over r and the reductions
        over x of its weight's and bias's gradients over [8192, 640] are one
        kernel, which reads x and g once (HelionScheduling.
        codegen_mix_order_reduction()): it loops over the x of each split of x,
        and sums the terms of each reduction over x into a partial, that the
        wrapper sums over the splits."""
        # On by default outside fbcode; Inductor marks the split (_split_size)
        # only with it on.
        self.enterContext(config.patch({"triton.mix_order_reduction": True}))
        xnumel, rnumel = 8192, 640
        x = torch.randn(xnumel, rnumel, device=DEVICE)
        g = torch.randn(xnumel, rnumel, device=DEVICE)
        w = torch.randn(rnumel, device=DEVICE)
        rstd = torch.rand(xnumel, 1, device=DEVICE) + 0.5

        def fn(x, g, w, rstd):
            gw = g * w
            dot = (gw * x).sum(-1, keepdim=True)
            dx = gw * rstd - x * rstd.pow(3) * dot / rnumel
            # A product over x too, the partial that starts at 1 and
            # multiplies (codegen_split_x()).
            return dx, (g * x * rstd).sum(0), g.sum(0), (1 + 1e-4 * g * x).prod(0)

        (source,) = self.compile(fn, x, g, w, rstd, tol=1e-3, backend=backend)
        self.assertRegex(source, r"for tile_y in hl.tile\(\d+\):")
        self.assertRegex(source, r"for tile_x in hl.tile\(\d+\):")
        self.assertIn("in0[tile_y, tile_x, :]", source)
        self.assertRegex(source, r"part(\d) = part\1 \+ torch\.sum\(")
        self.assertRegex(source, r"part(\d) = part\1 \* torch\.prod\(")
        self.assertRegex(source, r"ws\[2, tile_y, :\] = part2")
        self.assertIn(".sum(dim=0)", self.code)


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

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none", split_k_reduction="output")
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

    @parametrize(
        "native_matmul",
        [
            True,
            subtest(
                False,
                decorators=[
                    skipIfFbTriton(
                        "fbtriton's TLX plugin adds its matmul templates to "
                        "tuned_mm, and for this mm (a where() prologue into a "
                        "transposed b) their compute_epilogue reads "
                        "IterationRangesRoot.symt, which torch 2.16 does not "
                        "have: LoweringException: AttributeError, stock "
                        "Inductor's too (fbtriton 3.9.0.dev20261113, "
                        "fbtriton-h100 CI, Nov 2026)"
                    )
                ],
            ),
        ],
    )
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

    def test_persistent_dot(self):
        # A K of 32 is one block: one torch.mm of in0[tile_y, :] and
        # in1[:, tile_x], with no loop over tile_r and no accumulator. As in
        # test_saved_bf16_output, the bf16 product is saved and rounded, here
        # by torch.mm, and the epilogue reads it in fp32.
        fn = lambda a, b: torch.nn.functional.silu(a @ b)  # noqa: E731
        a, b = randn(64, 32), randn(32, 48).requires_grad_()
        (source,) = self.compile(fn, a, b, expected=fn(a, b), tol=2e-2)
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none", split_k_reduction="output")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([64, 48]):
        tmp0 = in0[tile_y, :]
        tmp1 = in1[:, tile_x]
        tmp2 = torch.mm(tmp0, tmp1)
        tmp3 = tmp2.to(torch.float32)
        tmp4 = -tmp3
        tmp5 = torch.exp(tmp4)
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tmp8 = tmp3 / tmp7
        out0[tile_y, tile_x] = tmp3.to(torch.bfloat16)
        out1[tile_y, tile_x] = tmp8.to(torch.bfloat16)
""",
        )

    def test_persistent_bmm(self):
        (source,) = self.compile(torch.bmm, randn(3, 40, 64), randn(3, 64, 136))
        self.assertIn("tmp2 = torch.bmm(tmp0, tmp1)", source)
        self.assertIn("out0[tile_z, tile_y, tile_x] = tmp2\n", source)
        self.assertNotIn("acc", source)

    def test_persistent_out_dtype(self):
        # An fp32 product of bf16 operands keeps its accumulator: torch.mm of the
        # bf16 tiles would round it to bf16.
        fn = lambda a, b: torch.mm(a, b, out_dtype=torch.float32)  # noqa: E731
        (source,) = self.compile(fn, randn(40, 32), randn(32, 136), tol=1e-3)
        self.assertIn("acc = torch.addmm(acc, tmp0, tmp1)", source)
        self.assertIn("out0[tile_y, tile_x] = acc\n", source)

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_split_k_reduction_follows_torch_flags(self, dtype):
        # A 16-bit dot takes torch's reduced-precision flags for its dtype, an fp32
        # output included; an fp32 dot has none.
        name = "bf16" if dtype is torch.bfloat16 else "fp16"
        attr = f"allow_{name}_reduced_precision_reduction"
        matmul = torch.backends.cuda.matmul
        saved = getattr(matmul, attr), getattr(matmul, f"{attr}_split_k")
        a = torch.randn(40, 72, device=DEVICE, dtype=dtype)
        b = torch.randn(72, 136, device=DEVICE, dtype=dtype)
        fp32_out = lambda a, b: torch.mm(a, b, out_dtype=torch.float32)  # noqa: E731
        try:
            for flags, expected in (
                ((True, True), "output"),
                ((False, True), "accumulator"),
                ((False, False), "none"),
            ):
                setattr(matmul, attr, flags)
                for fn in (torch.mm, fp32_out):
                    torch._dynamo.reset()
                    (source,) = self.compile(fn, a, b, tol=2e-2)
                    self.assertIn(
                        f'autotune_effort="none", split_k_reduction="{expected}")',
                        source,
                    )
        finally:
            setattr(matmul, attr, saved)
        (source,) = self.compile(torch.mm, a.float(), b.float(), tol=2e-2)
        self.assertNotIn("split_k_reduction", source)

    def test_persistent_dot_epilogue_reads_fp32(self):
        # The epilogue reads the bf16 product in fp32, as it would from memory. In
        # bf16, + 1000 would round it to a multiple of 4.
        fn = lambda a, b: (a @ b + 1000) - 1000  # noqa: E731
        (source,) = self.compile(fn, randn(64, 32), randn(32, 48), tol=2e-2)
        self.assertIn("tmp2 = torch.mm(tmp0, tmp1)", source)
        self.assertIn("tmp3 = tmp2.to(torch.float32)", source)

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

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none", split_k_reduction="output")
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
        tmp6 = 1.0
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
        self.assertIn("hl.load(in2, [tile_x], extra_mask=", source)
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
            # input would leave its storage, so the load is flat. (With the CuTe
            # backend, the pad is a kernel of its own: test_pad_prologue.)
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

    def test_operand_constant_fp32(self):
        # A float32 constant operand reaches the dot with no cast to make it a
        # tensor: HelionCSEProxy.dot() takes its 0-d tensor, which dot_operand()
        # expands (a Python float has no expand()).
        a = torch.randn(40, 72, device=DEVICE)
        b = torch.randn(72, 136, device=DEVICE)
        (source,) = self.compile(lambda a, b: a @ torch.ones_like(b), a, b, tol=1e-2)
        self.assertIn(".expand(tile_r, tile_x)", source)

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
        self.assertIn("torch.mm(", source)

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


def rotate_half_case():
    """RMSNorm then RoPE's rotate_half over head dims of 256, as Gemma's q/k
    norm, and its inputs: the cat cuts each persistent rdim at 128."""
    x = torch.randn(64, 8, 256, device=DEVICE)
    w = torch.randn(256, device=DEVICE)
    cos = torch.randn(64, 256, device=DEVICE)
    sin = torch.randn(64, 256, device=DEVICE)

    def fn(x, w, cos, sin):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
        rotated = torch.cat([-y[..., 128:], y[..., :128]], -1)
        return y * cos[:, None] + rotated * sin[:, None]

    return fn, (x, w, cos, sin)


def channel_cat_case():
    """GhostNet's cat of a passthrough x and relu(bn(y)) along the channels of
    channels-last tensors, and the mean of the cat over H and W (its
    squeeze-excitation pool): the cat cuts the channels in two equal parts."""
    x = torch.randn(64, 32, 8, 8, device=DEVICE).to(memory_format=torch.channels_last)
    y = torch.randn(64, 32, 8, 8, device=DEVICE).to(memory_format=torch.channels_last)
    w, b = torch.randn(32, device=DEVICE), torch.randn(32, device=DEVICE)

    def fn(x, y, w, b):
        out = torch.cat([x, torch.relu(y * w[:, None, None] + b[:, None, None])], 1)
        return out, out.mean([2, 3])

    return fn, (x, y, w, b)


def check_channel_cat_segments(test: InductorBackendTestCase, backend: str) -> None:
    """The cat's two parts are emitted as segments of the channel axis
    (HelionKernel.pointwise_segment()): the kernel tiles 32 channels, and
    reads and writes each part as a tile of its own parameter, with no
    select between them, where the whole axis of 64 read y flat behind a
    mask: hl.load(in1, [(-32) + tile_x.index + ...], extra_mask=...)."""
    fn, args = channel_cat_case()
    (source,) = test.compile(fn, *args, tol=1e-4, backend=backend)
    test.assertIn("for tile_y, tile_x in hl.tile([64, 32]):", source)
    test.assertNotIn("torch.where", source)
    # The parts' loads and stores are tiles; the unselected part of each
    # segment keeps a masked load, which nothing reads and Helion drops.
    test.assertRegex(source, r"tmp\d+ = in\d\[tile_y, :, tile_x\]\n")
    test.assertEqual(len(re.findall(r"out\d\[tile_y, :, tile_x\] = ", source)), 2)
    test.assertEqual(len(re.findall(r"out\d\[tile_y, tile_x\] = ", source)), 2)
    # Parts shorter than RDIM_SEGMENT_MIN stay one tile: torch.stack's four
    # parts of one element would turn the [1024, 4] tile into four stores of
    # every fourth element.
    a, b, c, d = (torch.randn(1024, device=DEVICE) for _ in range(4))
    (source,) = test.compile(
        lambda a, b, c, d: torch.stack([a, b, c, d], -1), a, b, c, d, backend=backend
    )
    test.assertNotIn("hl.tile([1024, 1])", source)


# The CuTe lane, where CuTe DSL is installed, runs this.
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestInductorCute(InductorBackendTestCase):
    @needs_native_matmul
    def test_bmm_epilogue(self):
        self.check_bmm_epilogue("cute")

    def test_mix_order_reduction(self):
        self.check_mix_order_reduction("cute")

    @needs_native_matmul
    def test_dot_keeps_k_loop(self):
        # A K of 64 is one block, but on an SM100-family GPU a 16-bit dot keeps its K
        # loop (MatmulPlan.keep_k_loop); elsewhere the persistent form stays.
        (source,) = self.compile(
            torch.bmm, randn(4, 128, 64), randn(4, 64, 128), backend="cute"
        )
        if torch.cuda.get_device_capability()[0] != 10:
            self.assertIn("tmp2 = torch.bmm(tmp0, tmp1)", source)
            return
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none", split_k_reduction="output")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([4, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = in0[tile_z, tile_y, tile_r]
            tmp1 = in1[tile_z, tile_r, tile_x]
            acc = torch.baddbmm(acc, tmp0, tmp1)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)
""",
        )
        (source,) = self.compile(
            torch.mm, randn(128, 64), randn(256, 64).t(), backend="cute"
        )
        self.assertIn("acc = torch.addmm(acc, tmp0, tmp1)", source)

    @needs_native_matmul
    def test_persistent_dot(self):
        # A small matmul the tcgen05 planner rejects keeps K in one block: a stored
        # torch.mm.
        (source,) = self.compile(torch.mm, randn(40, 64), randn(64, 96), backend="cute")
        self.assertIn("tmp2 = torch.mm(tmp0, tmp1)", source)
        self.assertNotIn("acc", source)
        # So does a batch of small matrices.
        (source,) = self.compile(
            torch.bmm, randn(256, 56, 32), randn(256, 32, 56), backend="cute"
        )
        self.assertIn("tmp2 = torch.bmm(tmp0, tmp1)", source)
        self.assertNotIn("acc", source)

    @needs_native_matmul
    def test_transposed_product(self):
        # A tall product over a column-major b lowers as (b.T @ a.T).T where the
        # pair-tile model says so (transposes()): the kernel reads views of the inputs
        # and writes an N-contiguous buffer whose transpose is the result. (A K of 4096
        # sums in another order than eager's.)
        (source,) = self.compile(
            torch.mm,
            randn(8192, 4096),
            randn(1024, 4096).t(),
            tol=1e-2,
            backend="cute",
        )
        # (each case checks its own rule's answer for the GPU at hand)
        if not transposes(8192, 1024, 4096):
            self.assertIn("for tile_y, tile_x in hl.tile([8192, 1024]):", source)
        else:
            # The dot's rows (tile_x, b's 1024 columns) are tiled first.
            self.assertIn("for tile_x, tile_y in hl.tile([1024, 8192]):", source)
            self.assertIn("tmp0 = in0[tile_x, tile_r]", source)
            self.assertIn("tmp1 = in1[tile_r, tile_y]", source)
            self.assertIn("out0[tile_x, tile_y] = acc.to(torch.bfloat16)", source)
            self.assertIn(
                "empty_strided_cuda((1024, 8192), (1, 1024), torch.bfloat16)",
                self.code,
            )
            self.assertIn(
                "helion_fused_mm_0(reinterpret_tensor(arg1_1, (1024, 4096), (4096, 1), 0), "
                "reinterpret_tensor(arg0_1, (4096, 8192), (1, 4096), 0), buf0)",
                self.code,
            )
            self.assertIn(
                "return (reinterpret_tensor(buf0, (8192, 1024), (1024, 1), 0), )",
                self.code,
            )
        # A one-wave product whose N no CTA pair tiles as written runs transposed.
        (source,) = self.compile(
            torch.mm, randn(512, 1280), randn(1000, 1280).t(), tol=1e-2, backend="cute"
        )
        if transposes(512, 1000, 1280):
            self.assertIn("for tile_x, tile_y in hl.tile([1000, 512]):", source)
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([512, 1000]):", source)
        # So does one whose N an aligned 128-row pair tiles finer transposed.
        (source,) = self.compile(
            torch.mm, randn(1000, 2048), randn(1024, 2048).t(), tol=1e-2, backend="cute"
        )
        if transposes(1000, 1024, 2048):
            self.assertIn("for tile_x, tile_y in hl.tile([1024, 1000]):", source)
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([1000, 1024]):", source)
        # So does a square-ish product of a row-major b, though not of a column-major
        # one.
        (source,) = self.compile(
            torch.mm, randn(2048, 2560), randn(2560, 2560), tol=1e-2, backend="cute"
        )
        if transposes(2048, 2560, 2560, b_n_major=True):
            self.assertIn("for tile_x, tile_y in hl.tile([2560, 2048]):", source)
            self.assertIn("tmp0 = in0[tile_x, tile_r]", source)
            self.assertIn(
                "reinterpret_tensor(arg1_1, (2560, 2560), (1, 2560), 0)", self.code
            )
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([2048, 2560]):", source)
        self.assertFalse(transposes(2048, 2560, 2560))
        # A wide product tiles finer as written, and the Triton backend keeps every
        # product as written.
        for backend, m, n in (("cute", 1000, 4096), ("triton", 8192, 1024)):
            (source,) = self.compile(
                torch.mm,
                randn(m, 4096),
                randn(n, 4096).t(),
                tol=1e-2,
                backend=backend,
            )
            self.assertIn(f"for tile_y, tile_x in hl.tile([{m}, {n}]):", source)

    @needs_native_matmul
    def test_transposed_product_by_rule(self):
        # A skinny product over a row-major b runs transposed
        # (tcgen05_transposes_skinny_product()), reading b.T as the dot's first operand.
        (source,) = self.compile(
            torch.mm, randn(128, 1000), randn(1000, 768), tol=1e-2, backend="cute"
        )
        if skinny_transposes(128, 768, 1000, b_n_major=True):
            self.assertIn("for tile_x, tile_y in hl.tile([768, 128]):", source)
            self.assertIn("tmp0 = in0[tile_x, tile_r]", source)
            self.assertIn(
                "reinterpret_tensor(arg1_1, (768, 1000), (1, 768), 0)", self.code
            )
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([128, 768]):", source)
        # A row-major b whose rows straddle L2 sectors runs transposed
        # (tcgen05_transposes_misaligned_b_rows()).
        (source,) = self.compile(
            torch.mm, randn(2048, 768), randn(768, 20008), tol=1e-2, backend="cute"
        )
        if misaligned_transposes(2048, 20008, 768):
            self.assertIn("for tile_x, tile_y in hl.tile([20008, 2048]):", source)
            self.assertIn(
                "reinterpret_tensor(arg1_1, (20008, 768), (1, 20008), 0)", self.code
            )
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([2048, 20008]):", source)
        # A wide, epilogue-bound product over a column-major b runs transposed
        # (tcgen05_transposes_epilogue_bound_product()).
        (source,) = self.compile(
            torch.mm,
            randn(2048, 512),
            randn(19712, 512).t(),
            tol=1e-2,
            backend="cute",
        )
        if epilogue_bound_transposes(2048, 19712, 512):
            self.assertIn("for tile_x, tile_y in hl.tile([19712, 2048]):", source)
            self.assertIn(
                "empty_strided_cuda((19712, 2048), (1, 19712), torch.bfloat16)",
                self.code,
            )
        else:
            self.assertIn("for tile_y, tile_x in hl.tile([2048, 19712]):", source)

    @needs_native_matmul
    def test_dot_axes_follow_the_output(self):
        # The dot's rows are the axis the output's store scales by its row stride,
        # whatever the operands load: ones_like(a) @ b keeps its rows on tile_y.
        a, b = randn(256, 512), randn(384, 512).t()
        (source,) = self.compile(
            lambda a, b: torch.ones_like(a) @ b, a, b, tol=1e-2, backend="cute"
        )
        self.assertIn("for tile_y, tile_x in hl.tile([256, 384]):", source)
        # The transposed product with a fused epilogue (an MLP's
        # relu(x @ w.t())) keeps its rows on tile_x and the epilogue in it.
        a, b = randn(4096, 1024), randn(1024, 1024).t()
        transposed = transposes(4096, 1024, 1024)
        (source,) = self.compile(
            lambda a, b: torch.relu(a @ b), a, b, tol=1e-2, backend="cute"
        )
        if transposed:
            self.assertIn("for tile_x, tile_y in hl.tile([1024, 4096]):", source)
        # a @ b + a @ c with c row-major transposes a @ b only, so the two dots stay in
        # kernels of their own (HelionScheduling.can_fuse()).
        # (small integers keep each bf16 product, and their sum, exact)
        a, b, c = (
            torch.randint(-1, 2, shape, device=DEVICE).to(torch.bfloat16)
            for shape in ((4096, 1024), (1024, 1024), (1024, 1024))
        )
        b = b.t()
        sources = self.compile(
            lambda a, b, c: a @ b + a @ c, a, b, c, tol=1e-2, backend="cute"
        )
        if transposed:
            self.assertEqual(len(sources), 2)
            self.assertTrue(
                any(
                    "for tile_x, tile_y in hl.tile([1024, 4096]):" in s for s in sources
                )
            )
            self.assertTrue(
                any(
                    "for tile_y, tile_x in hl.tile([4096, 1024]):" in s for s in sources
                )
            )
        # An operand that is still a prologue has no layout to prove K-major,
        # so its product stays as written.
        s = randn(1024, 1)
        # (the dot reads s + b rounded to bf16, as eager's bf16 sum is)
        expected = (a.float() @ (s + b).float()).to(torch.bfloat16)
        sources = self.compile(
            lambda a, b, s: a @ (s + b),
            a,
            b,
            s,
            expected=expected,
            tol=1e-2,
            backend="cute",
        )
        self.assertIn("for tile_y, tile_x in hl.tile([4096, 1024]):", sources[-1])

    @needs_native_matmul
    @config.patch(force_shape_pad=True)
    def test_padded_operand(self):
        # pad_mm pads N from 1001 to 1008. The pad is a kernel of its own, and
        # the dot reads the padded b as tiles, not through b's masked loads.
        a, b = randn(64, 128), randn(1001, 128).t()
        sources = self.compile(torch.matmul, a, b, backend="cute")
        self.assertEqual([bool(DOT_RE.search(s)) for s in sources], [False, True])
        self.assertIn("tmp1 = in1[tile_r, tile_x]", sources[1])
        self.assertNotIn("hl.load(", sources[1])

    @needs_native_matmul
    def test_pad_prologue(self):
        # F.pad of a is a kernel of its own too (on Triton it stays in the dot:
        # TestInductorMatmul.test_operand).
        fn = lambda a, b: torch.nn.functional.pad(a, (0, 8, 0, 4)) @ b  # noqa: E731
        sources = self.compile(fn, randn(36, 64), randn(72, 136), backend="cute")
        self.assertEqual([bool(DOT_RE.search(s)) for s in sources], [False, True])
        self.assertIn("tmp0 = in0[tile_y, tile_r]", sources[1])

    def test_unsupported_op(self):
        # Helion's CuTe backend compiles no lgamma.
        x = torch.rand(64, 100, device=DEVICE) + 0.5
        self.assertUnsupported("lgamma", torch.lgamma, x, backend="cute")
        (source,) = self.compile(torch.exp, x, tol=1e-4, backend="cute")
        self.assertIn('backend="cute"', source)

    @looped
    def test_arg_reduction(self):
        # arg_reduce()'s combine emits isnan, logical_and/or/not and an int64
        # where, which backend_ops() does not see in the node's body.
        x = torch.randn(64, 100, device=DEVICE)
        x[3, 10] = float("nan")
        fn = lambda x: (torch.argmax(x, -1), torch.any(x > 2.0, -1))  # noqa: E731
        (source,) = self.compile(fn, x, backend="cute")
        self.assertIn("torch.any(", source)

    @looped
    def test_online_softmax(self):
        # online_softmax_reduce()'s combine emits exp, abs, neg, maximum and
        # where, which backend_ops() does not see in the node's body.
        x = torch.randn(64, 100, device=DEVICE)
        fn = lambda x: torch.softmax(x, -1)  # noqa: E731
        (source,) = self.compile(fn, x, tol=1e-4, backend="cute")
        self.assertIn("torch.maximum(acc, ", source)

    def test_long_rdim_named_by_input(self):
        # TestInductorBackend.test_long_rdim_named_by_input's rdim of 4096,
        # named by the input's full rdim, whose load Helion drops.
        fn = lambda x: (x * torch.arange(4096, device=x.device)).sum(-1)  # noqa: E731
        x = torch.randn(64, 4096, device=DEVICE)
        (source,) = self.compile(fn, x, tol=1e-3, backend="cute")
        self.assertIn("rdim = in0[tile_x, :].size(1)", source)

    def test_expanded_input(self):
        # TestInductorBackend.test_expanded_input's stores of a value expanded
        # over the tiles its stride-0 input lacks, over 2-D and 3-D tiles.
        x = torch.randn(1, 7, 31, device=DEVICE).expand(64, 7, 31)
        (source,) = self.compile(lambda x: x * 2, x, backend="cute")
        self.assertIn("out0[tile_y, tile_x] = tmp2.expand(tile_y, tile_x)", source)
        i = torch.randint(0, 17, (1, 7), device=DEVICE).expand(64, 7)
        w = torch.randn(17, 31, device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            (source,) = self.compile(
                lambda i, w: F.embedding(i, w) * 2, i, w, backend="cute"
            )
        self.assertIn(".expand(tile_z, tile_y, tile_x)", source)

    def test_long_softmax_rdim(self):
        # Softmax over rdims of 4096 reads each rdim as one block, up to
        # PERSISTENT_SOFTMAX_RDIM: its max, its sum of exps and the output pass
        # over an rdim the autotuner can keep in registers, where Triton's loop
        # reads it twice (TestInductorBackend.test_long_softmax_rdim). A kernel
        # that only reduces the rdim, as cross-entropy's max and sum of exps,
        # keeps the loop, which reads it once (rereads_softmax_rdim()).
        x = torch.randn(64, 4096, device=DEVICE)
        fn = lambda x: torch.softmax(x, -1)  # noqa: E731
        (source,) = self.compile(fn, x, tol=1e-4, backend="cute")
        self.assertIn("tmp0 = in0[tile_x, :]", source)
        self.assertNotIn("for tile_r", source)
        t = torch.randint(0, 4096, (64,), device=DEVICE)
        fn = lambda x, t: F.cross_entropy(x, t, reduction="none")  # noqa: E731
        with config.patch(assert_indirect_indexing=False):
            sources = self.compile(fn, x, t, tol=1e-4, backend="cute")
        self.assertTrue(any("for tile_r in hl.tile(4096):" in s for s in sources))
        # So does one that also writes y = x * 2, which the softmax's results
        # do not reach: the loop writes it in the pass that reduces the rdim.
        (source,) = self.compile(
            self.logsumexp_beside_logits, x, tol=1e-4, backend="cute"
        )
        self.assertIn("for tile_r in hl.tile(4096):", source)

    @staticmethod
    def logsumexp_beside_logits(x):
        y = x * 2
        m = y.amax(-1, keepdim=True)
        return y, m + (y - m).exp().sum(-1, keepdim=True).log()

    def test_gathered_slices(self):
        # TestInductorBackend.test_gather_cases' embedding lookups: CuTe reads
        # w's rdim at each value as a view of w subscripted by the value
        # (gather_subscript()), in a reduction over the rdim and in a softmax
        # that reads it whole.
        w = torch.randn(1000, 64, device=DEVICE)
        i = torch.randint(0, 1000, (16384,), device=DEVICE)
        for fn in (
            lambda i, w: F.embedding(i, w).pow(2).sum(-1),
            lambda i, w: torch.softmax(F.embedding(i, w), -1),
        ):
            with config.patch(assert_indirect_indexing=False):
                (source,) = self.compile(fn, i, w, tol=1e-4, backend="cute")
            self.assertRegex(source, r"in1\[tmp\d+, :\]")

    def test_inplace_param(self):
        # TestInductorBackend.test_inplace_param under CuTe, whose one parameter
        # lets it reduce each rdim of 128 over 32 threads, four elements a
        # thread (num_threads=[1, 32]). With an input and an output passed buf1
        # each, that config fails with "synthetic-lane reduction would reorder
        # a potentially aliasing write".
        x = torch.randn(1, 1, 64, 128, device=DEVICE)
        w = torch.randn(1, 1, 3, 3, device=DEVICE)
        (source,) = self.compile(conv_inplace, x, w, tol=1e-4, backend="cute")
        self.assertIn("def kernel(in_out0: torch.Tensor, out0: torch.Tensor)", source)
        kernel = PyCodeCache.load(source).kernel
        conv = F.conv2d(x, w, padding=1).view(64, 128)
        y, mean = conv.clone(), torch.empty(64, device=DEVICE)
        warp_per_rdim = helion.Config(block_sizes=[1], num_threads=[1, 32])
        kernel.bind((y, mean)).compile_config(warp_per_rdim)(y, mean)
        torch.testing.assert_close(y, conv + 1)
        expected = (conv + 1).pow(2).mean(-1)
        torch.testing.assert_close(mean, expected, rtol=1e-4, atol=1e-4)

    def test_masked_view(self):
        # TestInductorBackend.test_masked_view's pad: CuTe reads x through its
        # [64, 250] view too, masked past 250 along an x of 256.
        x = torch.randn(64, 250, device=DEVICE)
        (source,) = self.compile(lambda x: F.pad(x, (0, 6)) * 2, x, backend="cute")
        self.assertRegex(source, MASKED_VIEW_LOAD)

    def test_repeat_kv_reads_the_quotient_as_a_view_dim(self):
        # repeat_kv: k, stored token-major [S, Hk, D], expanded to Hk * 2
        # heads of [S, D]. The output's three dims leave no tile for the
        # repeat, so each head reads the [Hk, S, D] view at its head // 2, a
        # run of D consecutive elements along x, rather than the flat storage at
        # x + 64*(z // 2) + 256*y.
        fn, args = repeat_kv_case()
        (source,) = self.compile(fn, *args, backend="cute")
        self.assertIn("in0[tile_z.index // 2, tile_y, tile_x]", source)

    def test_rotate_half_rdim_segments(self):
        # TestInductorBackend.test_rotate_half_rdim_segments under the CuTe
        # backend, which packs the halves' slices of the rdims into vector loads
        # and stores (one output parameter, segment_subscript()).
        fn, args = rotate_half_case()
        (source,) = self.compile(fn, *args, tol=1e-4, backend="cute")
        self.assertIn("out0[tile_y, tile_x, 0:128] = ", source)
        self.assertIn("out0[tile_y, tile_x, 128:256] = ", source)

    def test_channel_cat_segments(self):
        check_channel_cat_segments(self, "cute")

    @looped
    def test_welford(self):
        # welford()'s combine emits eq, where and truediv, and the block's weight
        # is the scalar tmp2 * (tile_r.end - tile_r.begin).
        x = torch.randn(64, 1000, device=DEVICE)
        (source,) = self.compile(lambda x: torch.var_mean(x, -1), x, backend="cute")
        self.assertIn("(tile_r.end - tile_r.begin)", source)

    def test_welford_combine(self):
        # A split var_mean over N, H and W of an NCHW [8, 64, 64, 64]
        # (HelionChoices.layout_split()): the second kernel's welford_combine
        # weighs the partials' means and keeps the weighted mean where its
        # correction is not finite, isinf(m) | isnan(m), which CuTe compiles.
        x = torch.randn(8, 64, 64, 64, device=DEVICE)
        fn = lambda x: torch.var_mean(x, [0, 2, 3], correction=0)  # noqa: E731
        _, second = self.compile(fn, x, tol=1e-3, backend="cute")
        for op in ("torch.isinf(", "torch.isnan(", "torch.logical_or("):
            self.assertIn(op, second)

    def test_persistent_reduction(self):
        # Rows of 300 in one block: softmax, argmax over hl.arange(rdim) and
        # var_mean.
        x = torch.randn(64, 300, device=DEVICE)

        def fn(x):
            return torch.softmax(x, -1), torch.argmax(x, -1), torch.var_mean(x, -1)

        sources = self.compile(fn, x, tol=1e-4, backend="cute")
        self.assertNotIn("for tile_r", "\n".join(sources))
        self.assertIn("hl.arange(rdim)", "\n".join(sources))

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


def repeat_kv_case():
    """repeat_kv, k stored token-major [S, Hk, D] expanded to Hk * 2 heads of
    [S, D], and its input."""

    def fn(k):
        s, hk, d = k.shape
        heads = k.permute(1, 0, 2)[:, None].expand(hk, 2, s, d)
        return heads.reshape(hk * 2, s, d) * 2

    return fn, (torch.randn(100, 4, 64, device=DEVICE),)


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

    @parametrize(
        "case",
        ["distinct", "repeated", "rows", "one_element", "constant"],
    )
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
                "constant": n * 0,
            }[case]
            out = torch.zeros(2 * x.size(0), *x.shape[1:], device=x.device)
            # constant: a 0-d value at an index with no tile, which each of
            # the 64 elements still adds once (HelionKernel.store()).
            value = torch.ones_like(x) if case == "constant" else x
            return out.index_put((index,), value, True)

        x = torch.randn(64, *([8] if case == "rows" else []), device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            sources = self.compile(fn, x, tol=1e-4)
        self.assertEqual(len([s for s in sources if "hl.atomic_add(" in s]), 1)
        if case == "constant":
            self.assertRegex(self.code, r"hl\.atomic_add\(\w+, \[tile_x\], tmp\d+\)")
        if case == "one_element":
            # Every x[i] adds at out[0], through a stride-0 view.
            self.assertRegex(
                self.code, r"reinterpret_tensor\(buf\d+, \(64, \), \(0, \), 0\)"
            )

    def test_keeps_cuda_custom_pass(self):
        # HelionPostGradPass runs the CUDA pass registered before it first.
        # _register() also sets the choices handler again, from every
        # registration and config.inductor_choices_class: under fbtriton that
        # composes its TLXInductorChoices ahead of HelionChoices, and the tests
        # after this one would read their rdims of 4096 in a loop,
        #     for tile_r in hl.tile(4096):
        #         tmp0 = in0[tile_x, tile_r]
        # where Helion's choice reads them whole (tmp0 = in0[tile_x, :]), so
        # the handler is restored.
        custom_pass = object()
        with (
            patch.dict(common.device_codegens),
            patch.dict(common.custom_backend_passes, {"cuda": custom_pass}),
            V.set_choices_handler(V.choices),
        ):
            helion.inductor._register()
            registered = common.custom_backend_passes["cuda"]
            self.assertIsInstance(registered, helion.inductor.HelionPostGradPass)
            self.assertIs(registered.inner, custom_pass)

    def test_options(self):
        # The settings are torch.compile options with a "helion." prefix.
        x = torch.randn(64, 100, device=DEVICE)
        options = {"cuda_backend": "helion", "helion.fast_math": True}
        with config.patch(cuda_backend="triton"):
            _, (code,) = run_and_get_code(torch.compile(torch.exp, options=options), x)
            self.assertIn("fast_math=True", code)
            with self.assertRaisesRegex(RuntimeError, "helion.fast_mathh"):
                torch.compile(torch.exp, options={"helion.fast_mathh": True})(x)

    @skipIfRefEager(
        "ref mode binds add_kernel without the host function Helion's Dynamo "
        "variable traces: kernel binding succeeded but host_function is None"
    )
    def test_helion_kernel_in_graph(self):
        # A @helion.kernel called in the compiled function is Helion's own
        # template, here with relu and * 2 fused into it (torch_compile_fusion).
        x, y = torch.randn(64, 128, device=DEVICE), torch.randn(64, 128, device=DEVICE)
        fn = lambda x, y: add_kernel(x, y).relu() * 2  # noqa: E731
        self.compile(fn, x, y, expected=fn(x, y), expect_helion=False)
        self.assertRegex(self.code, r"def _helion_add_\w*_fused_mul_relu_0\(")

    @skipIfRefEager(
        "ref mode runs scale_kernel in PyTorch: code_and_output returns its "
        "Python source, not Triton code"
    )
    def test_helion_kernel_numerics(self):
        # Helion lowers a @helion.kernel through Inductor as for cuda_backend
        # "triton", with the fp32 constant of x * 0.1 for a bf16 x.
        code, _ = code_and_output(scale_kernel, (randn(64, 128),))
        self.assertIn("tl.full([], 0.1, tl.float32)", code)

    @parametrize("case", ["online_softmax_off", "pattern_matcher_off", "integer"])
    @looped
    def test_softmax(self, case):
        # Without online softmax (turned off, or with Inductor's pattern
        # rewrites turned off, or for an integer x, whose exps
        # prepare_softmax_online would sum in an int64 accumulator), the max and
        # the sum are two reductions.
        # The sum's value derives from the max: exp(x - max), whose dtype and
        # shape HelionCSEProxy propagates.
        x = torch.randn(64, 1000, device=DEVICE)
        fn = lambda x: torch.softmax(x, -1)  # noqa: E731
        if case == "integer":
            x = torch.randint(-5, 5, (64, 1000), device=DEVICE)
            fn = lambda x: (  # noqa: E731
                (x - x.amax(-1, keepdim=True)).exp().sum(-1, keepdim=True)
            )
        settings = {
            "online_softmax_off": {"online_softmax": False},
            "pattern_matcher_off": {"pattern_matcher": False},
            "integer": {},
        }[case]
        with config.patch(settings):
            (source,) = self.compile(fn, x, tol=1e-4)
        self.assertIn("acc1 = acc1 + torch.sum(", source)

    @looped
    def test_online_softmax(self):
        # helion.inductor's post-grad pass rewrites softmax to
        # prepare_softmax_online, as Inductor's does for Triton only: one pass
        # over each row for its max and its sum of exps, then one for the output.
        (source,) = self.compile(
            lambda x: torch.softmax(x, -1), torch.randn(64, 100, device=DEVICE)
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(64):
        acc = hl.full([tile_x], float("-inf"), dtype=torch.float32)
        acc1 = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(100):
            tmp0 = in0[tile_x, tile_r]
            tmp1 = float("-inf")
            tmp2 = 0.0
            tmp3 = torch.amax(tmp0, -1)
            tmp4 = tmp3.reshape([tile_x, 1])
            tmp5 = tmp0 - tmp4
            tmp6 = torch.exp(tmp5)
            tmp7 = tmp3 == tmp1
            tmp8 = tmp7.reshape([tile_x, 1])
            tmp9 = torch.where(tmp8, tmp2, tmp6)
            tmp10 = torch.sum(tmp9, -1)
            tmp11 = torch.maximum(acc, tmp3)
            tmp12 = acc - tmp3
            tmp13 = torch.abs(tmp12)
            tmp14 = -tmp13
            tmp15 = torch.exp(tmp14)
            tmp16 = 1.0
            tmp17 = acc == tmp1
            tmp18 = acc == tmp11
            tmp19 = torch.where(tmp18, tmp16, tmp15)
            tmp20 = acc1 * tmp19
            tmp21 = torch.where(tmp17, tmp2, tmp20)
            tmp22 = tmp3 == tmp11
            tmp23 = torch.where(tmp22, tmp16, tmp15)
            tmp24 = tmp10 * tmp23
            tmp25 = tmp11 == tmp1
            tmp26 = float("nan")
            tmp27 = tmp21 + tmp24
            tmp28 = torch.where(tmp25, tmp26, tmp27)
            acc1 = tmp28
            acc = tmp11
        for tile_r in hl.tile(100):
            tmp29 = in0[tile_x, tile_r]
            tmp30 = acc.reshape([tile_x, 1])
            tmp31 = tmp29 - tmp30
            tmp32 = torch.exp(tmp31)
            tmp33 = acc1.reshape([tile_x, 1])
            tmp34 = tmp32 / tmp33
            out0[tile_x, tile_r] = tmp34
""",
        )

    @looped
    def test_online_softmax_sum(self):
        # The bare sum of a row of only -inf is nan, as in eager:
        # exp(-inf - -inf). Its -inf terms are 0, and the sum is nan where the
        # max is -inf.
        x = torch.randn(4, 300, device=DEVICE)
        x[1] = float("-inf")
        fn = lambda x: (  # noqa: E731
            (x - x.amax(-1, keepdim=True)).exp().sum(-1, keepdim=True)
        )
        (source,) = self.compile(fn, x, tol=1e-4)
        self.assertRegex(source, r"acc1 = tmp\d+\n")

    def test_online_softmax_looped(self):
        # Blocks of 64 over rows whose first half is -inf: the accumulators hold
        # (-inf, nan) when the first finite block arrives, whose nan sum the
        # combine leaves out (nan * 0 would keep it).
        x = torch.randn(4, 4096, device=DEVICE)
        x[:, :2048] = float("-inf")
        pinned = helion.Config(block_sizes=[1, 64, 64])
        with patch("helion.kernel", functools.partial(helion.kernel, config=pinned)):
            self.compile(lambda x: torch.softmax(x, -1), x, tol=1e-4)

    @parametrize("case", ["split", "neg_inf_row", "log_softmax"])
    @looped
    def test_online_softmax_cases(self, case):
        # split: the second kernel of a split reduction reduces the first's
        # (max, sum) pairs, and the first reduces blocks of the -inf half, and
        # (-inf, 0) past the end of R (65537 elements, a prime: no split divides
        # it);
        # neg_inf_row: softmax of an rdim of -inf is nan, as in eager;
        # log_softmax: the same rewrite, x - max - log(sum).
        if case == "split":
            x = torch.randn(8, 65537, device=DEVICE)
            x[:, :32768] = float("-inf")
        else:
            x = torch.randn(4, 300, device=DEVICE)
            x[1] = float("-inf")
        fn = torch.log_softmax if case == "log_softmax" else torch.softmax
        sources = self.compile(lambda x: fn(x, -1), x, tol=1e-4)
        self.assertEqual(len(sources), 3 if case == "split" else 1)
        # The sum's accumulator takes the combine, not acc1 + torch.sum(...).
        self.assertRegex("\n".join(sources), r"acc1 = tmp\d+\n")

    @parametrize("op", ["cumsum", "sort"])
    def test_aten_fallback(self, op):
        # No BackendFeature.SCAN or SORT: Inductor lowers them to aten kernels.
        fn = {
            "cumsum": lambda x: torch.cumsum(x, -1) * 2,
            "sort": lambda x: torch.sort(x, -1).values * 2,
        }[op]
        (source,) = self.compile(fn, torch.randn(64, 100, device=DEVICE))
        self.assertRegex(self.code, rf"aten\.{op}\.\w+\(")

    @looped
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
            tmp1 = 2.0
            tmp2 = tmp0 * tmp1
            acc = acc + torch.sum(tmp2, -1)
        tmp3 = in1[tile_x]
        tmp4 = acc + tmp3
        out0[tile_x] = tmp4
""",
        )

    def test_persistent_reduction_source(self):
        # An amax and a sum are one block, a persistent reduction, at any
        # length (HelionChoices.should_use_persistent_reduction()): x is read
        # once, and exp(x - max) computed once for both the sum and the output.
        # In a loop over tile_r, three loops read x and the last two compute
        # the exp.
        def fn(x):
            e = torch.exp(x - x.amax(-1, keepdim=True)) * 2
            return e / e.sum(-1, keepdim=True)

        (source,) = self.compile(fn, torch.randn(64, 300, device=DEVICE))
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(64):
        tmp0 = in0[tile_x, :]
        tmp1 = torch.amax(tmp0, -1)
        tmp2 = tmp1.reshape([tile_x, 1])
        tmp3 = tmp0 - tmp2
        tmp4 = torch.exp(tmp3)
        tmp5 = 2.0
        tmp6 = tmp4 * tmp5
        tmp7 = torch.sum(tmp6, -1)
        tmp8 = tmp7.reshape([tile_x, 1])
        tmp9 = tmp6 / tmp8
        out0[tile_x, :] = tmp9
""",
        )

    @parametrize(
        "case",
        ["argmax", "pow2", "few", "welford", "neg_inf_row", "broadcast", "pad"],
    )
    def test_persistent_reduction(self, case):
        # argmax: the row's indices are hl.arange(rdim), with rdim the dim of
        # a factory's dim, hl.zeros([tile_x, 300]), as an rdim of up to 1024
        # stays one block (HelionKernel.codegen_rdim()); a NaN, ties and an rdim
        # of -inf as in test_arg_reduction. pow2: an rdim of 128 is named rdim
        # too, not 128. few: an rdim of 12, not a power of two, keeps a loop of one
        # block, even under choices that make every reduction persistent
        # (names_rdim()). welford: rdims in one block each, whose weight is R:
        # test_welford_extremes' rdims of inf and its rdim of 1e38 and 0 in turn,
        # whose sum overflows, so that its mean is inf, as from Inductor's
        # persistent reductions (sum(x) / weight), where eager's Welford keeps
        # it finite, and rdims of equal values (3.0 and 1000.0) whose sums are
        # exact in a block of 963, whose variance is 0.
        # neg_inf_row: softmax of an rdim of -inf is nan, as in eager. broadcast:
        # the max over a broadcast dim of z, all -0.0, is -0.0. pad: rows of
        # 304 whose last 4 the mask (hl.arange(rdim) < 300) takes from the pad.
        x = torch.randn(64, 300, device=DEVICE)
        fn = lambda x: (torch.argmax(x, -1), torch.argmin(x, -1))  # noqa: E731
        expected = None
        if case == "argmax":
            x[3, 10] = x[3, 250] = x[5, 200] = float("nan")
            x[2] = float("-inf")
            x[7, 50:] = 1.0
        elif case == "pow2":
            x = torch.randn(64, 128, device=DEVICE)
        elif case == "few":
            x = torch.randn(64, 12, device=DEVICE)
            self.enterContext(V.set_choices_handler(PersistentChoices()))
        elif case == "welford":
            x = torch.randn(6, 963, device=DEVICE)
            x[1], x[2, 480:] = float("inf"), float("inf")
            x[3], x[4], x[5] = 3.0, 1000.0, 0.0
            x[5, ::2] = 1e38
            fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
            var, mean = torch.var_mean(x, -1)
            mean[1:3] = mean[5] = float("inf")
            expected = (var, mean)
        elif case == "neg_inf_row":
            x[1] = float("-inf")
            fn = lambda x: torch.softmax(x, -1)  # noqa: E731
        elif case == "broadcast":
            x = torch.full((64, 1), -0.0, device=DEVICE)
            fn = lambda z: torch.signbit(torch.amax(z.expand(64, 300), -1))  # noqa: E731
        elif case == "pad":
            fn = lambda x: torch.softmax(F.pad(x, (0, 4), value=-1.0), -1)  # noqa: E731
        (source,) = self.compile(fn, x, expected=expected, tol=1e-4)
        # One block per row: no loop, and no accumulators across blocks.
        self.assertEqual("for tile_r" in source, case == "few")
        self.assertEqual("acc" in source, case == "few")
        line = {
            "argmax": "rdim = hl.zeros([tile_x, 300]).size(1)",
            "pow2": "hl.arange(rdim)",
            "few": "for tile_r in hl.tile(12):",
            "welford": " = 963.0\n",
            "neg_inf_row": "tmp0 = in0[tile_x, :]",
            "broadcast": ".expand(tile_x, rdim)",
            "pad": "[hl.arange(rdim) + 300*tile_x.index.reshape([tile_x, 1])]",
        }[case]
        self.assertIn(line, source)
        self.assertEqual(
            "rdim" in source, case in ("argmax", "pow2", "broadcast", "pad")
        )

    def test_long_rdim_reduction(self):
        # A sum over an rdim of 4096, past Triton's 1024 for a persistent rdim,
        # reads the rdim as one block, which Helion can roll into a loop
        # (Config.reduction_loops); var_mean keeps its loop, which combines
        # each block into the Welford accumulators as it reads it.
        x = torch.randn(64, 4096, device=DEVICE)
        (source,) = self.compile(lambda x: x.sum(-1), x, tol=1e-3)
        self.assertIn("tmp0 = in0[tile_x, :]", source)
        self.assertNotIn("for tile_r", source)
        (source,) = self.compile(lambda x: torch.var_mean(x, -1), x, tol=1e-4)
        self.assertIn("for tile_r in hl.tile(4096):", source)
        # So does argmax, whose one block reduces the rdim twice (its max, then
        # the first index of it), where the loop folds both in one pass.
        (source,) = self.compile(lambda x: torch.argmax(x, -1), x)
        self.assertIn("for tile_r in hl.tile(4096):", source)

    def test_long_softmax_rdim(self):
        # Softmax over rdims of 4096 keeps the online-softmax loop, which reads
        # each rdim twice, where CuTe reads each rdim as one block
        # (TestInductorCute.test_long_softmax_rdim).
        x = torch.randn(64, 4096, device=DEVICE)
        (source,) = self.compile(lambda x: torch.softmax(x, -1), x, tol=1e-4)
        self.assertIn("for tile_r in hl.tile(4096):", source)

    def test_long_unnamed_rdim(self):
        # An rdim of 2**21, unsplit, that only a flipped (flat) load reads: no
        # full rdim of an operand names its dim, and a factory of 2**21 does not
        # compile, so it keeps a loop over tile_r
        # (HelionScheduling.create_kernel_choices()).
        x = torch.randn(2, 2**21, device=DEVICE)
        with config.patch(split_reductions=False):
            (source,) = self.compile(lambda x: x.flip(-1).sum(-1), x, tol=1e-2)
        self.assertIn("for tile_r in hl.tile(2097152):", source)
        # A flipped rdim of 100000, past UNROLLED_RDIM_MAX: no input's full rdim
        # names it, and a factory's dim would make each rdim one block of
        # 131072 lanes, so it loops over tile_r too.
        x = torch.randn(4, 100000, device=DEVICE)
        with config.patch(split_reductions=False):
            (source,) = self.compile(lambda x: x.flip(-1).sum(-1), x, tol=1e-2)
        self.assertIn("for tile_r in hl.tile(100000):", source)

    def test_split_reduction_loops(self):
        # The first kernel of a split reduction keeps Triton's rule
        # (HelionChoices.should_use_persistent_reduction(), which reads
        # Inductor's _split_size): the chunks of x.sum(0) over [100000, 96],
        # outer reductions of more than 64 values each, loop over tile_r.
        # Inductor marks the split only with mix_order_reduction on.
        self.enterContext(config.patch({"triton.mix_order_reduction": True}))
        x = torch.randn(100000, 96, device=DEVICE)
        first, _ = self.compile(lambda x: x.sum(0), x, tol=1e-3)
        self.assertIn("for tile_r in hl.tile(", first)

    def test_long_rdim_named_by_input(self):
        # A persistent rdim past UNROLLED_RDIM_MAX that the kernel names, for the
        # arange it broadcasts over the rdim, takes the dim of an input's full
        # rdim (HelionKernel.codegen_rdim()), which the autotuner can roll:
        #     rdim = in0[tile_x, :].size(1)
        #     tmp1 = (hl.arange(rdim)).to(torch.float32)
        # The sum agrees with eager in one block, and rolled into blocks of 512
        # (on an x of 32: a new kernel source, which Helion compiles again).
        fn = lambda x: (x * torch.arange(4096, device=x.device)).sum(-1)  # noqa: E731
        (source,) = self.compile(fn, torch.randn(64, 4096, device=DEVICE), tol=1e-3)
        self.assertIn("rdim = in0[tile_x, :].size(1)", source)
        kernel, configs = helion.kernel, []

        def rolled(fn=None, **kwargs):
            if fn is None:  # @helion.kernel(backend=..., ...)
                return functools.partial(rolled, **kwargs)
            configs.append(helion.Config(block_sizes=[1], reduction_loops=[512]))
            return kernel(fn, config=configs[-1], **kwargs)

        with patch("helion.kernel", rolled):
            self.compile(fn, torch.randn(32, 4096, device=DEVICE), tol=1e-3)
        self.assertEqual(len(configs), 1)

    def test_inspected_kernel_counts_once(self):
        # A persistent rdim past UNROLLED_RDIM_MAX is emitted once to tell what
        # it needs of its rdim (HelionScheduling.create_kernel_choices()), then
        # again: metrics.generated_kernel_count, which tests compare with the
        # kernels a graph launches, counts it as it counts an rdim of 512, which
        # is emitted once.
        counts = []
        for n in (512, 4096):
            metrics.reset()
            with config.patch(fx_graph_cache=False):
                self.compile(
                    lambda x: x.sum(-1), torch.randn(16, n, device=DEVICE), tol=1e-4
                )
            counts.append(metrics.generated_kernel_count)
        self.assertEqual(counts[1], counts[0])

    def test_permuted_copy_and_epilogue(self):
        # y = x.permute(0, 2, 3, 1).contiguous() and y * 2: one kernel, whose
        # tiling is natural_tiling()'s for each node and for the pair alike
        # (HelionScheduling.can_fuse()), writes both.
        x = torch.randn(8, 64, 32, 32, device=DEVICE)

        def fn(x):
            y = x.permute(0, 2, 3, 1).contiguous()
            return y, y * 2

        self.assertEqual(len(self.compile(fn, x)), 1)

    @parametrize("case", ["mean", "broadcast", "mask"])
    def test_natural_tiling(self, case):
        # Tiles over the dims the indexes divide the flat ranges by, with no
        # // or % (natural_tiling()). mean: the mean over H and W of a
        # channels-last x, read at 64*r + 2304*(x // 64) + x % 64 by the
        # 1-D tiling, an outer reduction, whose contiguous C is the last dim of
        # the tile. broadcast: w[c] beside x[n, c], read at x % 384.
        # mask: the x and y, x % 100 and x // 100, of a causal mask.
        if case == "mean":
            x = torch.randn(8, 64, 6, 6, device=DEVICE)
            x = x.to(memory_format=torch.channels_last)
            (source,) = self.compile(lambda x: x.mean([2, 3]), x, tol=1e-4)
            self.assertIn("for tile_y, tile_x in hl.tile([8, 64]):", source)
            self.assertIn("in0[tile_y, :, tile_x]", source)
            self.assertIn("torch.sum(tmp0, -2)", source)
        elif case == "broadcast":
            x = torch.randn(32, 384, device=DEVICE)
            w = torch.randn(384, device=DEVICE)
            (source,) = self.compile(lambda x, w: x * w + 1, x, w)
            self.assertIn("for tile_y, tile_x in hl.tile([32, 384]):", source)
            self.assertIn("in1[tile_x]", source)
        else:
            x = torch.randn(100, 100, device=DEVICE)

            def fn(x):
                i = torch.arange(100, device=x.device)
                return torch.where(i[None, :] <= i[:, None], x, 0.0)

            (source,) = self.compile(fn, x)
            self.assertIn("for tile_y, tile_x in hl.tile([100, 100]):", source)
        self.assertNotRegex(source, r"tile_\w\.index[^\n]*(//|%)")

    def test_mix_order_reduction(self):
        # Pin blocks [1, 8]. The default [32, 16] (A10G, A100) loads [32, 16, 1024]
        # tiles that take Triton and ptxas over 14 minutes, past CI's 180 s timeout.
        with pin_blocks(8):
            self.check_mix_order_reduction("triton")

    def test_mix_order_reduction_off(self):
        # config.triton.mix_order_reduction off: the sum over x is a kernel of
        # its own, as for Triton.
        x, g = (
            torch.randn(8192, 640, device=DEVICE),
            torch.randn(8192, 640, device=DEVICE),
        )
        with config.patch({"triton.mix_order_reduction": False}):
            sources = self.compile(
                lambda x, g: ((g * x).sum(-1), (g * x).sum(0)), x, g, tol=1e-3
            )
        self.assertNotIn("ws[", "".join(sources))

    @parametrize("op", ["sum", "prod"])
    @looped
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
    @looped
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

    @looped
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
            tmp1 = 0.0
            tmp2 = 1.0
            tmp3 = tmp2 * (tile_r.end - tile_r.begin)
            tmp4 = tmp2 / tile_r.block_size
            tmp5 = tmp0 * tmp4
            tmp6 = torch.sum(tmp5, -1)
            tmp7 = hl.full([], tmp3, dtype=torch.float32)
            tmp8 = tmp7 * tmp4
            tmp9 = tmp6 / tmp8
            tmp10 = tmp9.reshape([tile_x, 1])
            tmp11 = tmp0 - tmp10
            tmp12 = tmp11 * tmp11
            tmp13 = torch.sum(tmp12, -1)
            tmp14 = acc2 + tmp3
            tmp15 = acc == tmp9
            tmp16 = tmp9 - acc
            tmp17 = torch.where(tmp15, tmp1, tmp16)
            tmp18 = tmp14 == tmp1
            tmp19 = tmp7 / tmp14
            tmp20 = torch.where(tmp18, tmp1, tmp19)
            tmp21 = tmp17 * tmp20
            tmp22 = acc + tmp21
            tmp23 = acc1 + tmp13
            tmp24 = acc2 * tmp20
            tmp25 = tmp17 * tmp24
            tmp26 = tmp25 * tmp17
            tmp27 = tmp23 + tmp26
            acc = tmp22
            acc1 = tmp27
            acc2 = tmp14
        out0[tile_x] = acc
        tmp28 = 99.0
        tmp29 = acc1 / tmp28
        out1[tile_x] = tmp29
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
        # as close to eager's as Triton's; rows 8-15 of 2**123 (1.1e37), whose
        # sum overflows, and whose scaled values add up exactly in any order
        # (of 1e37, they round in the order fbtriton's and ref mode's sums take,
        # and the deviations from that mean square to inf in the variance);
        # rows 16-23 of 100000000 and 100000008 in turn, whose mean
        # rounds to 100000000 or 100000008: the deviations 0 and 8 give the
        # variance 32 * 4096 / 4095 as from Inductor's persistent reductions,
        # where eager gives half of it.
        fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
        if case == "split":
            x = torch.randn(8, 65537, device=DEVICE) + 100
            self.assertEqual(len(self.compile(fn, x)), 2)
            # Halves of 1e37 and -1e37 over the masked split: partial means of
            # 1e37 with weights of thousands, whose product overflows. The mean
            # is finite and the variance overflows, as in eager.
            x[:, :32768], x[:, 32768:] = 1e37, -1e37
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            var, mean = compiled(x)
            self.assertTrue(torch.isinf(var).all() and torch.isfinite(mean).all())
            # One value of 1e5 among randn(): the combine weighs the partials'
            # means and corrects that, where deviations from the largest mean
            # would each round by its spread (the mean moved by 5e-4).
            y = torch.randn(8, 65537, device=DEVICE)
            y[:, 7] = 1e5
            _, mean = compiled(y)
            expected = y.double().mean(-1).float()
            torch.testing.assert_close(mean, expected, rtol=0, atol=1e-5)
            # Halves of 3e38 and -3e38, in chunks of one sign each (65536 = 2**16
            # values): partial means of 3e38 with weights of thousands, whose
            # product overflows. The mean is finite and the variance overflows, as in
            # eager. (A chunk of both signs would overflow the merge of its
            # blocks' means, delta = -6e38, as Triton's welford_combine does.)
            x = torch.empty(8, 65536, device=DEVICE)
            x[:, :32768], x[:, 32768:] = 3e38, -3e38
            var, mean = torch.compile(fn, fullgraph=True, dynamic=False)(x)
            self.assertTrue(torch.isinf(var).all() and torch.isfinite(mean).all())
            return
        x = torch.randn(24, 4096, device=DEVICE)
        x[:8, 100], x[8:16], x[16:] = 2000.0, 2.0**123, 100000000.0
        x[16:, ::2] = 100000008.0
        var, mean = fn(x)
        var[16:] = 32 * 4096 / 4095
        self.assertEqual(len(self.compile(fn, x, expected=(var, mean))), 1)

    def test_welford_constant_rdim(self):
        # Constant rdims whose mean rounds, of one block each: the persistent
        # Welford takes the mean as sum(x) / n, as Inductor's persistent
        # reductions do, and the squared deviations from it, which leave a
        # variance of at most an ulp of the value squared where eager's running
        # mean gives 0 (stock Inductor gives 2.3e23 for the rdim of 2e19 and 9.9
        # for the rdim of 100553424).
        x = torch.empty(2, 963, device=DEVICE)
        x[0], x[1] = 2e19, 100553424.0
        var, mean = torch.compile(lambda x: torch.var_mean(x, -1), fullgraph=True)(x)
        torch.testing.assert_close(mean, x[:, 0], rtol=0, atol=0)
        ulp = torch.tensor([2.0**41, 8.0], device=DEVICE)
        self.assertTrue(((var >= 0) & (var <= ulp * ulp)).all(), var)

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

    def test_sink_softmax(self):
        # Softmax with attention sinks over rdims of 100 scores and a sink,
        # whose slice is made contiguous, as before the matmul with v (a slice
        # returned as it is keeps the cat: needs_cat_storage()): the
        # reductions over the cat dim are the scores' and the sink's,
        # combined (split_cat_reductions()), so one kernel reads each rdim of
        # scores as a tile and writes it normalized.
        scores = torch.randn(4, 32, 100, device=DEVICE)
        sinks = torch.randn(4, device=DEVICE)

        def fn(scores, sinks):
            x = torch.cat([scores, sinks[:, None, None].expand(-1, 32, 1)], -1)
            x = x - x.amax(-1, keepdim=True)
            return torch.softmax(x, -1)[..., :-1].contiguous()

        (source,) = self.compile(fn, scores, sinks, tol=1e-4)
        self.assertIn("in0[tile_y, tile_x, :]", source)
        self.assertIn("torch.maximum(", source)
        self.assertNotIn("extra_mask", source)

    def test_proven_mask(self):
        # (cat([x, y], -1) * 2)[:, :50] reads only x, inside its part, which
        # split_cat_reductions() leaves to the kernel: the value ranges prove
        # tile_x.index < 100 over its [64, 50] tiles, so x's load has no mask
        # (HelionKernel.masked()) and no torch.where picks between x and y
        # (HelionOverrides.where()).
        x = torch.randn(64, 100, device=DEVICE)
        y = torch.randn(64, 4, device=DEVICE)
        fn = lambda x, y: (torch.cat([x, y], -1) * 2)[:, :50].exp()  # noqa: E731
        (source,) = self.compile(fn, x, y)
        self.assertIn("tmp5 = in0[tile_y, tile_x]", source)
        self.assertNotIn("torch.where", source)
        # A cat along the first dim, whose mask over y reaches
        # torch.where reshaped to [tile_y, 1] (HelionKernel.align_args()).
        fn = lambda x, y: (torch.cat([x, y], 0) * 2)[:32].exp()  # noqa: E731
        (source,) = self.compile(fn, x, torch.randn(4, 100, device=DEVICE))
        self.assertNotIn("torch.where", source)

    def test_pad_rdim(self):
        # Rdims of 2000 and 3000, past UNROLLED_RDIM_MAX, of which an input holds
        # 1900 and 2100: rdim is the whole rdim's, not that input's, which
        # would drop the rest of it.
        x = -torch.ones(8, 1900, device=DEVICE)
        (source,) = self.compile(lambda x: F.pad(x, (0, 100), value=1.0).sum(-1), x)
        self.assertNotIn("rdim = in0[tile_x, :].size(1)", source)
        a, b = torch.ones(8, 2100, device=DEVICE), torch.ones(8, 900, device=DEVICE)
        self.compile(lambda a, b: torch.cat([a, b], -1).sum(-1), a, b)

    def test_nchw_reductions(self):
        # sum and var_mean over N, H and W of an NCHW tensor split into a
        # chunk per run of H * W contiguous values (HelionChoices.layout_split()):
        # the first kernel reads each run as an rdim of a view, with no division
        # or remainder in its index, and the second combines the 8 partials of
        # each channel. 2M values, past MIN_LAYOUT_SPLIT_VALUES, split whether
        # or not Inductor would.
        x = torch.randn(8, 64, 64, 64, device=DEVICE)
        for fn in (
            lambda x: x.sum([0, 2, 3]),
            lambda x: torch.var_mean(x, [0, 2, 3], correction=0),
        ):
            first, second = self.compile(fn, x, tol=1e-3)
            self.assertRegex(first, r"in0\[tile_x, (tile_r|:)\]")
            self.assertNotIn("//", first)
            self.assertIn("in0[tile_x, :]", second)
        # A batch of 2, whose 2 runs per channel Inductor's 16 chunks cut in 8
        # each: chunks within a run, still a tile of a view.
        x = torch.randn(2, 64, 256, 256, device=DEVICE)
        first, _ = self.compile(lambda x: x.mean((0, 2, 3)), x, tol=1e-3)
        self.assertIn("in0[tile_z, tile_y, tile_x, ", first)
        self.assertNotIn("//", first)
        # 8192 values, which Inductor does not split: one kernel, whose launch
        # a second kernel's would double.
        small = torch.randn(2, 64, 8, 8, device=DEVICE)
        self.assertEqual(len(self.compile(lambda x: x.sum([0, 2, 3]), small)), 1)

    @parametrize("case", ["persistent", "looped"])
    def test_outer_reductions(self, case):
        # Reductions over the outer dim of [1000, 100] (Inductor's hint OUTER),
        # whose kernels read rdims of x (reads_strided_rdim()) and reduce the rdim
        # before tile_x, in one block or a loop over tile_r, as the sums of
        # test_nchw_reductions: argmax and argmin with a NaN, which keep
        # Triton's rule (LOOPED_REDUCTIONS: an outer rdim past 64 loops), and
        # var_mean, amax and prod, which Inductor splits, whose first kernel
        # reads in0[tile_y, tile_r, tile_x] and whose second combines the
        # partials.
        # Inductor marks the split's first kernel (_split_size), which keeps
        # Triton's rule and loops, only with mix_order_reduction on.
        self.enterContext(config.patch({"triton.mix_order_reduction": True}))
        x = torch.randn(1000, 100, device=DEVICE)
        x[10, 3] = float("nan")
        rdim = ":" if case == "persistent" else "tile_r"
        for fn, coordinate in (
            (lambda x: (torch.argmax(x, 0), torch.argmin(x, 0)), "tile_r"),
            (lambda x: torch.var_mean(x, 0), rdim),
            (lambda x: (x.amax(0), x.prod(0).abs().clamp(max=1.0)), rdim),
        ):
            with config.patch({"triton.persistent_reductions": case == "persistent"}):
                sources = self.compile(fn, x, tol=1e-4)
            for source in sources:
                self.assertRegex(
                    source, rf"in0\[(tile_y, tile_r|{coordinate}), tile_x\]"
                )

    def test_outer_sum(self):
        # A sum over y with an x too large to split, which Inductor hints
        # DEFAULT: its loads read more bytes with a unit stride along x than
        # along r0_ (tiling.contiguous_reads()), so the rdim is the first dim
        # of the tile and x, of stride 1, the last (reads_strided_rdim()), not
        # in0[tile_x, :].
        x = torch.randn(64, 32768, device=DEVICE)
        (source,) = self.compile(lambda x: x.sum(0), x, tol=1e-4)
        self.assertIn("tmp0 = in0[:, tile_x]", source)
        self.assertIn("torch.sum(tmp0, -2)", source)

    @skipIfRefEager("reference mode sequences nothing: it searches no config")
    def test_kernels_tuned_in_sequence(self):
        # The two kernels of a split sum over y: the second one's first call
        # is timed behind the first's (helion.inductor.kernel_sequence),
        #     helion_fused_sum_1 = PyCodeCache.load(helion_fused_sum_1_source, "<key>").kernel
        #     helion_fused_sum_1 = helion_in_sequence(
        #         helion_fused_sum_1, globals(), "helion_fused_sum_1", after="helion_fused_sum_0"
        #     )
        preludes = []
        timed_behind = kernel_sequence.timed_behind
        self.enterContext(
            patch.object(
                kernel_sequence,
                "timed_behind",
                lambda prelude: preludes.append(prelude) or timed_behind(prelude),
            )
        )
        # The sequence on Triton, whose searches read CUDA events and so tune
        # alone (times_short_kernels, which test_inductor_kernel_sequence.py
        # checks): taken as if they read the trace, as CuTe's do.
        self.enterContext(
            patch.object(kernel_sequence, "times_short_kernels", lambda bound: True)
        )
        self.enterContext(config.patch({"triton.mix_order_reduction": True}))
        x = torch.randn(100000, 96, device=DEVICE)
        self.assertEqual(len(self.compile(lambda x: x.sum(0), x, tol=1e-3)), 2)
        # The second kernel names the first, whose call is just before its
        # (helion.inductor.codegen.sequence); the first names none.
        definitions = re.findall(
            r'(\w+) = helion_in_sequence\(\1, globals\(\), "\1", after=([^)]*)\)',
            self.code,
        )
        self.assertEqual(len(definitions), 2)
        (first, first_after), (second, second_after) = definitions
        self.assertEqual((first_after, second_after), ("None", repr(first)))
        # The second loads a module of its own, under the first's source.
        self.assertIn(f"{first} = PyCodeCache.load({first}_source).kernel", self.code)
        self.assertRegex(
            self.code,
            rf"{second} = PyCodeCache.load\({second}_source, '[0-9a-f]{{16}}'\).kernel",
        )
        self.assertEqual([prelude is None for prelude in preludes], [True, False])
        self.assertEqual(len(preludes[1].steps), 1)
        # Off, the kernels are called as they are loaded.
        with helion_config.patch(tune_in_sequence=False):
            torch._dynamo.reset()
            self.compile(lambda x: x.sum(0), x, tol=1e-3)
        self.assertNotIn("helion_in_sequence", self.code)

    def test_split_reduction_chunks(self):
        # Inductor cuts x.sum(0), an rdim of 100000, into chunks by a
        # factor of the GPU (782 chunks of 128 on a B200 or an H100), the last
        # past the end of the rdim, which masks every load of the first
        # kernel; HelionChoices.reduction_split_factor() picks a divisor of
        # the rdim near it (800 chunks of 125 there), whose loads are tiles of
        # a view of x, with x's contiguous dim last (an outer reduction).
        x = torch.randn(100000, 96, device=DEVICE)
        # Inductor marks the split (_split_size) only with mix_order_reduction
        # on, its default outside fbcode.
        self.enterContext(config.patch({"triton.mix_order_reduction": True}))
        # The sums of 100000 values, in chunks, differ from eager's by 1e-4.
        first, _ = self.compile(lambda x: x.sum(0), x, tol=1e-3)
        chunks = re.search(r"for tile_y, tile_x in hl\.tile\(\[(\d+), 96\]\):", first)
        chunk = re.search(r"for tile_r in hl\.tile\((\d+)\):", first)
        assert chunks is not None and chunk is not None, first
        self.assertEqual(int(chunks.group(1)) * int(chunk.group(1)), 100000)
        self.assertNotIn("extra_mask", first)
        self.assertIn("tmp0 = in0[tile_y, tile_r, tile_x]", first)

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

    @parametrize(
        "case",
        [
            subtest(
                "rows",
                decorators=[
                    skipIfRefEager(
                        "ref mode runs each hl.tile as one block over its whole "
                        "range, not the pinned blocks of 64: the block scale is "
                        "1 / 963, not 1 / 64, so row 3's mean of 2e19 rounds "
                        "and its deviations give the variance 4.8e24, not 0"
                    )
                ],
            ),
            "split",
            "split_empty",
        ],
    )
    @looped
    def test_welford_extremes(self, case):
        # Blocks of tile_r pinned to 64, or 1 for split_empty (by default one
        # block covers a row). rows, R = 963, whose last block has 3 lanes:
        # rows 1 and 2, of inf, get the mean inf, as from Triton's persistent
        # reductions (eager gives nan), since the delta of two equal means is
        # 0, where inf - inf is nan. Row 3, of 2e19, gets the variance 0, as on
        # its first block acc_weight is 0 and delta 2e19, whose square is inf in
        # float32. Rdim 4, of 100553424 (c): the last block's sum of 3c rounds,
        # so its mean, sum(x) / 3 as Inductor's persistent reductions take it,
        # is 100553432, and its deviations of 8 give the variance (3 * 64 + 64 *
        # 960 * 3 / 963) / 962, where eager gives 0. Rdim 5, of 1e38 and 0 in
        # turn, gets the mean 5e37: its deviations of 5e37 are scaled before
        # they are summed. split: test_welford_cases' masked split, whose first
        # kernel has partials of weight 0 and mean 0 past the end of R: rdims of
        # 2e19, whose squared deviation from that 0 is inf, get the variance 0,
        # and so does rdim 0, of -134217720, whose chunks' equal means the
        # combine keeps (their max equals their min), where their weighted sum
        # rounds and gives the variance 64.9.
        # split_empty: the same split, whose last block of the last chunk holds
        # only a partial of weight 0, whose mean would be 0 / 0, and whose 0
        # would make the mean of rdim 1, of inf, nan in the merge.
        rows = case == "rows"
        if rows:
            x = torch.randn(6, 963, device=DEVICE)
            x[1], x[2, 480:] = float("inf"), float("inf")
            x[3], x[4], x[5] = 2e19, 100553424.0, 0.0
            x[5, ::2] = 1e38
        elif case == "split":
            x = torch.randn(8, 65537, device=DEVICE) * 3 + 100
            x[0], x[4:] = -134217720.0, 2e19
        else:
            x = torch.randn(8, 65537, device=DEVICE)
            x[1] = float("inf")
        var, mean = torch.var_mean(x, -1)
        if rows:
            mean[1:3] = float("inf")
            var[4] = (3 * 64 + 64 * 960 * 3 / 963) / 962
        elif case == "split_empty":
            mean[1] = float("inf")
        with pin_blocks(1 if case == "split_empty" else 64):
            fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
            sources = self.compile(fn, x, expected=(var, mean))
        self.assertEqual(len(sources), 1 if rows else 2)

    def test_cast_round_trip(self):
        # A bool widened to int64 and cast back is the bool (exact_cast()):
        # torch.any reads the comparison, where each lane would widen to int64
        # and back.
        x = torch.randn(64, 300, device=DEVICE)
        (source,) = self.compile(lambda x: (x > 0).long().bool().any(-1), x)
        self.assertNotIn(".to(torch.bool)", source)
        # float8_e8m0fnu has no 0: False casts to 2**-127 and back to True. A
        # bf16 value widened to float32 is not folded back: under compute
        # types its cast to bf16 is the float32 value itself.
        self.assertFalse(exact_cast(torch.bool, torch.float8_e8m0fnu))
        self.assertTrue(exact_cast(torch.bool, torch.float8_e4m3fn))
        self.assertTrue(exact_cast(torch.int16, torch.float32))
        self.assertFalse(exact_cast(torch.int32, torch.float32))
        self.assertFalse(exact_cast(torch.bfloat16, torch.float32))

    def test_mean_beside_var(self):
        # torch.mean(x, -1) beside torch.var(x, -1) in one block: the kernel
        # sums x first, torch.sum(tmp0, -1), and the Welford block takes that
        # block sum: two block sums, the other m2's. A loop over tile_r keeps
        # the sum's own, in either order: three (a loop's sums accumulate in
        # acc lines, which the CSE does not hold). Rdim 0, of 1e37, whose sum
        # overflows: its mean is inf, as in eager, and so is its variance
        # where the Welford block takes the graph's sum, as from Inductor's
        # persistent reductions (eager gives 0).
        x = torch.randn(64, 1024, device=DEVICE)
        x[0] = 1e37
        mean_var = lambda x: (x.mean(-1), x.var(-1))  # noqa: E731
        mean, var = mean_var(x)
        var[0] = float("inf")
        (source,) = self.compile(mean_var, x, expected=(mean, var), tol=1e-4)
        self.assertIn("torch.sum(tmp0, -1)", source)
        self.assertEqual(source.count("torch.sum("), 2)
        x[0] = 1.0
        for fn in (mean_var, lambda x: (x.var(-1), x.mean(-1))):
            with looped:
                (source,) = self.compile(fn, x, tol=1e-4)
            self.assertEqual(source.count("torch.sum("), 3)

    def test_welford_share(self):
        # An rdim of 41 values of 3.0 in a block of 64: the merge's share, the
        # block's weight over the accumulated weight, divides a 0-d tensor of
        # the weight, 41 / 41 = 1, where Tensor.__rtruediv__ of the number
        # would take 1 / 41 * 41 = 0.99999994, and the mean 2.9999998:
        #     tmp7 = hl.full([], tmp3, dtype=torch.float32)
        #     tmp19 = tmp7 / tmp14
        # rather than tmp19 = tmp3 / tmp14. The block's mean, sum(x * s) /
        # (weight * s), is a float32 division, which the backend need not
        # round exactly (2 ulp without fast_math), so the values are checked to
        # 5e-7, absolute and relative (about 8 ulp of 3.0), and the share, the
        # fix, by the code.
        x = torch.full((8, 41), 3.0, device=DEVICE)
        with pin_blocks(64), looped, config.patch(split_reductions=False):
            (source,) = self.compile(lambda x: torch.var_mean(x, -1), x, tol=5e-7)
        weight = re.search(
            r"(tmp\d+) = hl\.full\(\[\], tmp\d+, dtype=torch\.float32\)", source
        )
        self.assertIsNotNone(weight)
        self.assertRegex(source, rf"= {weight[1]} / tmp\d+\n")
        # In float64 the weight and the scale are 0-d tensors of
        # hl.full([], 1.0, dtype=torch.float64), not numbers to wrap again in
        # hl.full([], tmp3, dtype=torch.float64).
        with pin_blocks(64), looped, config.patch(split_reductions=False):
            (source,) = self.compile(lambda x: torch.var_mean(x, -1), x.double())
        self.assertNotRegex(source, r"hl\.full\(\[\], tmp")

    @skipIfRefEager(
        "ref mode runs each hl.tile as one block over its whole range, so the "
        "block scale is 1 / 65536, not the pinned 1 / 64, and x * s is "
        "subnormal: the mean 4.4e-5 off"
    )
    def test_welford_small_values(self):
        # A loop over blocks of 64 of rdims of 65536 values of 1.0005e-36: each
        # block is scaled by 1 / 64, its size, so x * s stays normal, where
        # 1 / 65536, R's, would make it subnormal (1.5e-41), which keeps some
        # 13 bits (the mean 4.4e-5 off) or flushes to 0.
        x = torch.full((8, 65536), 1.0005e-36, device=DEVICE)
        with pin_blocks(64), looped, config.patch(split_reductions=False):
            var, mean = torch.compile(torch.var_mean, fullgraph=True)(x, -1)
        torch.testing.assert_close(mean, x.mean(-1), rtol=1e-5, atol=0)

    @parametrize("case", ["persistent", "looped"])
    def test_welford_large_mean(self, case):
        # Rdims whose squared deviations overflow float32, as eager's float32
        # Welford does: [2**87] * 15 + [2**87 + 2**64], whose mean rounds to
        # 2**87, so that its deviation of 2**64 squared is inf (its m2, 3.19e38,
        # would fit), and 1e38 * rand(16), 1e37 in a persistent rdim, whose mean
        # is the unscaled sum(x) / 16 (HelionKernel.welford()), inf where the
        # sum overflows. Their variance is inf, not nan, and their mean finite.
        # looped: blocks of 8.
        x = torch.full((4, 16), 2.0**87, device=DEVICE)
        x[:2, -1] = 2.0**87 + 2.0**64
        scale = 1e37 if case == "persistent" else 1e38
        x[2:] = scale * torch.rand(2, 16, device=DEVICE)
        fn = lambda x: torch.var_mean(x, -1)  # noqa: E731
        if case == "persistent":
            self.compile(fn, x)
        else:
            with pin_blocks(8), looped:
                self.compile(fn, x)

    def test_masked_view(self):
        # A pad of x from 250 to 256: tile_x.index < 250 masks the last
        # 6 lanes, which bounds tile_x to 250 in x's load (masked_extents()),
        # so it reads x through its own [64, 250] view, tiled as the output,
        #     tmp5 = hl.load(in0, [tile_y, tile_x], extra_mask=tmp4)
        # rather than at the flat index tile_x.index + 250*tile_y.index[:, None].
        x = torch.randn(64, 250, device=DEVICE)
        (source,) = self.compile(lambda x: F.pad(x, (0, 6)) * 2, x)
        self.assertRegex(source, MASKED_VIEW_LOAD)

    def test_where_wrapped_cast(self):
        # int8(arange(256))[128:] < 0 holds on every lane, as the cast wraps
        # 128..255 to -128..-1, where Inductor's value ranges of the condition
        # say it never does: the torch.where keeps a (HelionKernel.proven()).
        def fn(a):
            wrapped = torch.arange(256, dtype=torch.int8, device=a.device)[128:]
            return torch.where(wrapped < 0, a, 0.0)

        self.compile(fn, torch.randn(128, device=DEVICE))

    def test_masked_view_wrapped_cast(self):
        # A mask of an index cast to int8 bounds no tile: int8(tile_x.index) <
        # 100 holds at 128..255 too, where the cast wraps negative, so y is
        # read through its own [256] view, not a [100] one whose lanes past 100
        # the mask leaves on.
        def fn(y):
            mask = torch.arange(256, dtype=torch.int8, device=y.device) < 100
            index = torch.arange(256, device=y.device)
            return torch.ops.aten._unsafe_masked_index(y, mask, [index], 0.0) * 2

        (source,) = self.compile(fn, torch.randn(256, device=DEVICE))
        self.assertIn("hl.load(in0, [tile_x], extra_mask=", source)
        self.assertNotIn("(100, ), (1, )", self.code)

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
    for tile_y, tile_x in hl.tile([64, 10]):
        tmp0 = in0[tile_y, tile_x]
        tmp1 = tmp0.to(torch.int64)
        tmp2 = in1[tmp1 + 100*tile_y.index.reshape([tile_y, 1])]
        tmp3 = 2.0
        tmp4 = tmp2 * tmp3
        out0[tile_y, tile_x] = tmp4
""",
        )

    @parametrize(
        "case",
        [
            "embedding",
            "embedding_norm",
            "embedding_softmax",
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
        # One case per lowering of an indirect index, and embedding_softmax for
        # a gathered rdim that the online softmax reads whole (13 > 8).
        # embedding_norm:
        # w's rdims read at i[tile_x] in a reduction over them, as a view of
        # w at the value, in1[tmp1, :] (gather_subscript()), as index_select's
        # slices are, in1[tmp5, tile_x]: with an x of 16384, enough for Inductor's
        # hint DEFAULT, an inner reduction, whose gathered rdims are contiguous
        # along it (contiguous_reads()). embedding_softmax: such rdims through an
        # online softmax to a store of the whole rdim. maxpool_offsets: max_pool2d's backward
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
                lambda i, w: torch.nn.functional.embedding(i, w).pow(2).sum(-1),
                (torch.randint(0, 1000, (16384,), device=DEVICE), w),
            ),
            "embedding_softmax": (
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
            sources = self.compile(fn, *args, tol=1e-4)
        self.assertNotIn("@triton.jit", self.code)
        views = {
            "embedding_norm": r"in1\[tmp\d+, :\]",
            "embedding_softmax": r"in1\[tmp\d+, :\]",
            "index_select": r"in1\[tmp\d+, tile_x\]",
        }
        if case in views:
            self.assertRegex("\n".join(sources), views[case])

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

    @looped
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

    def test_inplace_param(self):
        # The conv's output, which the kernel adds 1 to and stores again (buf1
        # takes over buf0), is one parameter that the kernel reads and writes
        # (kernel_params()), passed once, not an input and an output passed
        # buf1 each: kernel(in_out0, out0), not kernel(in0, out0, out1).
        x = torch.randn(1, 1, 64, 128, device=DEVICE)
        w = torch.randn(1, 1, 3, 3, device=DEVICE)
        (source,) = self.compile(conv_inplace, x, w, tol=1e-4)
        self.assertIn("def kernel(in_out0: torch.Tensor, out0: torch.Tensor)", source)
        self.assertRegex(source, r"= in_out0\[tile_x, :\]")
        self.assertRegex(source, r"in_out0\[tile_x, :\] = ")

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
        if case == "two_seeds":
            # Both seeds are loads that need no computed value: they come ahead
            # of the computation, before the first draw.
            seeds = [m.start() for m in re.finditer(r"= in\d+\[\(\)\]", source)]
            self.assertEqual(len(seeds), 2)
            self.assertLess(max(seeds), source.index("hl.rand("))

    def test_number_over_tensor(self):
        # 1e-39 / x divides a 0-d tensor of the number by x: as a Python
        # number, Python would evaluate it as x.__rtruediv__(1e-39), which
        # Helion traces as a reciprocal times 1e-39, inf where x is 1e-39 (its
        # reciprocal overflows float32).
        x = torch.full((64, 128), 1e-39, device=DEVICE)
        (source,) = self.compile(lambda x: torch.full_like(x, 1e-39) / x, x)
        self.assertRegex(source, r"= hl\.full\(\[\], tmp\d+, dtype=torch\.float32\)")

    def test_constants_keep_their_dtype(self):
        # A float32 constant is a Python number beside a tensor, which rounds it
        # to float32 as eager does. Of constants alone the operator is a tensor
        # one: 16777217.0 == 16777216.0 in float32, not in Python's doubles.
        # Inductor folds a comparison of two constants before codegen, so the
        # test turns that folding off for the kernel to compare them.
        x = torch.randn(8, 16, device=DEVICE)
        with config.patch(joint_graph_constant_folding=False):
            (source,) = self.compile(
                lambda x: torch.full_like(x, 16777217.0).eq(16777216.0), x
            )
        self.assertIn("hl.full([], tmp", source)
        # A float64 constant stays a float64 tensor: the 0.1 of clamp_min is not
        # rounded to float32.
        y = torch.rand(8, 16, device=DEVICE, dtype=torch.float64) * 0.2
        (source,) = self.compile(lambda y: torch.clamp_min(y, 0.1), y, tol=0)
        self.assertIn("dtype=torch.float64", source)

    def test_rotate_half_rdim_segments(self):
        # RMSNorm then RoPE over head dims of 256, as Gemma's q/k norm: the cat
        # of rotate_half cuts the persistent rdim at 128, and the elementwise
        # tail is emitted once per half (HelionKernel.rdim_segment()), which
        # reads and writes slices of the rdims, as the hand-written kernel does,
        # where the whole rdim read the halves as flat loads masked by r < 128
        # and selected between them. (The part a half proves unselected keeps its
        # masked loads, which nothing reads and Helion drops.)
        fn, args = rotate_half_case()
        (source,) = self.compile(fn, *args, tol=1e-4)
        self.assertNotIn("torch.where", source)
        # Each half is a slice of the whole rdim's parameters (segment_subscript()),
        # whose tries take no rdim where they decline: the kernel names no rdim.
        # (The unselected part's masked load reads a flat view of x, in2.)
        self.assertIn("[tile_y, tile_x, 128:256]", source)
        self.assertIn("out0[tile_y, tile_x, 0:128] = ", source)
        self.assertIn("out0[tile_y, tile_x, 128:256] = ", source)
        self.assertNotIn("rdim", source)

    def test_channel_cat_segments(self):
        check_channel_cat_segments(self, "triton")

    def test_long_rdims_join_dims_split_by_r_invariant_reads(self):
        # An rdim of 2048 over (y, x) = [4, 16], with an r-invariant value of
        # a [4, 17] tensor sliced [:, 1:], whose dims do not join: y and x are
        # tiled as one axis of 64 (natural_tiling(), R_INVARIANT_JOIN_MIN),
        # and the r-invariant value is read flat, once per (y, x).
        x = torch.randn(4, 16, 2048, device=DEVICE)
        m = torch.randn(4, 17, device=DEVICE)
        fn = lambda x, m: (x - m[:, 1:, None]).exp().sum(-1)  # noqa: E731
        (source,) = self.compile(fn, x, m, tol=1e-4)
        self.assertIn("for tile_x in hl.tile(64):", source)
        self.assertRegex(source, r"in\d\[1 \+ 17\*\(tile_x\.index // 16\)")
        # An rdim of 256 keeps both pointwise dims.
        x = torch.randn(4, 16, 256, device=DEVICE)
        (source,) = self.compile(fn, x, m, tol=1e-4)
        self.assertIn("for tile_y, tile_x in hl.tile([4, 16]):", source)

    def test_written_before_keeps_the_whole_rdim(self):
        # z = x * 2, an output an earlier node of the kernel writes, which the
        # trailing nodes read beside a cat: a segment of them would read the
        # whole rdim's z (the value the CSE keeps of its store), so the rdim is
        # not cut (HelionScheduling.rdim_segments()).
        x, w = (torch.randn(64, 8, 256, device=DEVICE) for _ in range(2))
        sin = torch.randn(64, 256, device=DEVICE)

        def fn(x, w, sin):
            z = x * 2
            norm = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
            rotated = torch.cat([-w[..., 128:], w[..., :128]], -1)
            return z, norm + rotated * sin[:, None]

        (source,) = self.compile(fn, x, w, sin, tol=1e-4)
        self.assertNotIn("128:256]", source)

    def test_long_rdim_keeps_one_tail(self):
        # rotate_half over rdims of 2048, past UNROLLED_RDIM_MAX: the tail is one
        # emission over the whole rdim, which Helion can roll, not segments of
        # 1024, whose literal dims it cannot (HelionKernel.rdim_code()).
        x = torch.randn(16, 2048, device=DEVICE)

        def fn(x):
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            return torch.cat([-y[..., 1024:], y[..., :1024]], -1)

        (source,) = self.compile(fn, x, tol=1e-4)
        self.assertIn("rdim = in0[tile_x, :].size(1)", source)
        self.assertNotIn("0:1024", source)

    def test_rotate_half_rdim_segments_in_place(self):
        # The tail adds rotate_half of an RMSNorm of w into x in place: the
        # node reads x and stores x, so that each half must read its own x,
        # not the half before's x stored through the CSE store cache.
        x = torch.randn(64, 8, 256, device=DEVICE)
        w = torch.randn(64, 8, 256, device=DEVICE)
        cos = torch.randn(64, 256, device=DEVICE)
        sin = torch.randn(64, 256, device=DEVICE)

        def fn(x, w, cos, sin):
            y = w * torch.rsqrt(w.pow(2).mean(-1, keepdim=True) + 1e-6)
            rotated = torch.cat([-y[..., 128:], y[..., :128]], -1)
            return x.add_(y * cos[:, None] + rotated * sin[:, None])

        expected = fn(x.clone(), w, cos, sin)
        (source,) = self.compile(
            fn, x.clone(), w, cos, sin, expected=expected, tol=1e-4
        )
        self.assertIn("[tile_y, tile_x, 128:256]", source)

    def test_rotate_half_rdim_segments_of_a_wider_buffer(self):
        # As test_rotate_half_rdim_segments, of rdims that are the first 256
        # of a matmul's rdims of 768 (q of a fused qkv projection): the part
        # of the cat the second half proves unselected reads r = 256-383,
        # which the view of the matmul's buffer holds, so that its
        # masked load reads a slice too,
        #     hl.load(in0, [tile_y, tile_x, slice(0, 128)], extra_mask=tmp44)
        # where in0[tile_y, tile_x, 0:128] in an index list is a SyntaxError.
        a = torch.randn(512, 64, device=DEVICE)
        wqkv = torch.randn(64, 768, device=DEVICE)
        w = torch.randn(256, device=DEVICE)
        cos = torch.randn(64, 256, device=DEVICE)
        sin = torch.randn(64, 256, device=DEVICE)

        def fn(a, wqkv, w, cos, sin):
            x = (a @ wqkv).view(64, 8, 768)[..., :256]
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
            rotated = torch.cat([-y[..., 128:], y[..., :128]], -1)
            return y * cos[:, None] + rotated * sin[:, None]

        sources = self.compile(fn, a, wqkv, w, cos, sin, tol=1e-3)
        self.assertTrue(any("slice(0, 128)" in source for source in sources))

    def test_reduced_dim_quotient(self):
        # A reduced dim's quotient (w.repeat_interleave(2) read at r // 2 in a
        # persistent rdim) keeps the flat read: quotient_subscript() takes a
        # tile's only.
        x = torch.randn(64, 256, device=DEVICE)
        w = torch.randn(128, device=DEVICE)
        fn = lambda x, w: (x * w.repeat_interleave(2)).sum(-1)  # noqa: E731
        (source,) = self.compile(fn, x, w, tol=1e-4)
        self.assertIn("// 2", source)
        self.assertNotIn("tile_r.index // 2", source)

    def test_repeat_kv_reads_the_quotient_as_a_view_dim(self):
        # TestInductorCute's test under the Triton backend: each head reads
        # the [Hk, S, D] view at its head // 2 (quotient_subscript()).
        fn, args = repeat_kv_case()
        (source,) = self.compile(fn, *args)
        self.assertIn("in0[tile_z.index // 2, tile_y, tile_x]", source)

    def test_constant_index(self):
        # upsample_nearest over an output dim of size 1 reads a constant index,
        # index_expr(0, torch.float32): a Python number, which its cast takes as
        # a tensor.
        x = torch.randn(2, 3, 1, 8, device=DEVICE)
        self.compile(lambda x: F.interpolate(x, size=(1, 16), mode="nearest"), x)

    @pytest.mark.slow
    @skipIfLowVRAM("two masks of 2 GiB", required_bytes=8 * 2**30)
    @skipIfRefEager(
        "ref mode materializes hl.rand's 2**31 + 16 int64 offsets at once: "
        "CUDA out of memory for 16 GiB on CI's A10G"
    )
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

    @looped
    def test_broadcast_reduction(self):
        # The max over a broadcast dim of z, all -0.0, is -0.0, as in eager.
        z = torch.full((64, 1), -0.0, device=DEVICE)
        fn = lambda z: torch.signbit(torch.amax(z.expand(64, 300), -1))  # noqa: E731
        (source,) = self.compile(fn, z)
        self.assertIn(".expand(tile_x, tile_r)", source)

    def test_expanded_input(self):
        # A graph input expanded along a dim (stride 0) reads no tile of that
        # dim, and the store expands the value over it: x.expand(64, 7, 31) * 2
        # (its tiles [64, 217]), and an embedding of expanded indices.
        x = torch.randn(1, 7, 31, device=DEVICE).expand(64, 7, 31)
        (source,) = self.compile(lambda x: x * 2, x)
        self.assertIn("out0[tile_y, tile_x] = tmp2.expand(tile_y, tile_x)", source)
        i = torch.randint(0, 17, (1, 7), device=DEVICE).expand(64, 7)
        w = torch.randn(17, 31, device=DEVICE)
        with config.patch(assert_indirect_indexing=False):
            (source,) = self.compile(lambda i, w: F.embedding(i, w) * 2, i, w)
        self.assertIn(".expand(tile_z, tile_y, tile_x)", source)
        # A 0-d value stores as it is: torch.zeros(256) is out0[tile_x] = tmp0
        # of tmp0 = hl.full([], 0.0, ...).
        fn = lambda x: (x * 2, torch.zeros(256, device=DEVICE))  # noqa: E731
        sources = self.compile(fn, torch.randn(64, 256, device=DEVICE))
        (fill,) = [source for source in sources if "in0" not in source]
        self.assertRegex(fill, r"out0\[tile_x\] = tmp\d+\n")
        # Nor does a value align() gives the store's rank: each rdim's max,
        # stored over that rdim, out0[tile_x, :] = tmp3.reshape([tile_x, 1]).
        fn = lambda x: x.amax(-1, keepdim=True).expand_as(x) * 2  # noqa: E731
        (source,) = self.compile(fn, torch.randn(64, 256, device=DEVICE))
        self.assertRegex(
            source, r"out0\[tile_x, :\] = tmp\d+\.reshape\(\[tile_x, 1\]\)\n"
        )

    def test_natural_tiling_joins(self):
        # natural_tiling() joins dims past three pointwise and one reduction
        # dim by the cheapest joins. A 5-D permuted copy keeps three pointwise
        # dims, reading the joined ones through // and %:
        #     for tile_z, tile_y, tile_x in hl.tile([384, 6, 10]):
        # and a sum over dims 1 and 3 one reduction dim, an rdim of 16 * 64:
        #     for tile_y, tile_x in hl.tile([8, 32]):
        #         ... 2048*(hl.arange(rdim) // 64) ... ((hl.arange(rdim) % 64))
        x = torch.randn(4, 6, 8, 10, 12, device=DEVICE)
        (source,) = self.compile(lambda x: x.permute(0, 2, 4, 1, 3).contiguous(), x)
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([384, 6, 10]):", source)
        x = torch.randn(8, 16, 32, 64, device=DEVICE)
        (source,) = self.compile(lambda x: x.sum((1, 3)), x, tol=1e-4)
        self.assertIn("for tile_y, tile_x in hl.tile([8, 32]):", source)
        self.assertIn("(hl.arange(rdim) // 64)", source)

    @config.patch({"triton.codegen_upcast_to_fp32": False})
    @looped
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
