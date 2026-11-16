from __future__ import annotations

import importlib
import re
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen import common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
import torch.utils._pytree as pytree

from helion._compat import requires_torch_version
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA

if not requires_torch_version("2.16"):
    # helion.inductor uses the Inductor of torch 2.16, e.g. config.strict_reduction.
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

import helion.inductor
from helion.inductor import config as helion_config

SOURCE_RE = re.compile(r'helion_\w+_source = """\\\n(.*?)"""', re.DOTALL)
DOT_RE = re.compile(r"torch\.b?addb?mm\(")


# Inductor lowers matmuls natively for a backend through
# BaseScheduling.lowers_native_matmul (pytorch/pytorch#204025).
needs_native_matmul = unittest.skipUnless(
    "lowers_native_matmul" in vars(BaseScheduling),
    "torch without BaseScheduling.lowers_native_matmul",
)


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device=DEVICE, dtype=dtype)


class InductorBackendTestCase(TestCase):
    def setUp(self):
        super().setUp()
        helion.inductor.enable()
        self.addCleanup(helion.inductor.disable)
        patches = {"native_matmul": True, "autotune_effort": "none"}
        self.enterContext(helion_config.patch(patches))

    def compile(
        self, fn, *args, expected=None, tol=None, backend="triton", expect_helion=True
    ):
        """Run compiled `fn`, check it against FP32 eager, and return the Helion
        kernel sources."""
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
        sources = SOURCE_RE.findall(self.code)
        if expect_helion:
            self.assertTrue(sources)
            self.assertNotRegex(self.code, r"extern_kernels\.\w*mm\w*\(")
            self.assertNotIn("tl.dot(", self.code)
        return sources

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

    @parametrize(
        "case",
        [
            "none",
            "dynamic",
            "deterministic",
            "native_matmul_off",
            "gathered_operand",
            "cpu_scalar",
        ],
    )
    def test_matmul_fallback(self, case):
        a, b = randn(32, 64), randn(64, 48)
        fn = lambda a, b: torch.relu(a @ b)  # noqa: E731
        args = [a, b]
        # Declined when lowering: the matmul keeps its extern kernel, as without
        # Helion, rather than a Triton native matmul kernel.
        lowering = "extern_kernels.mm("
        if case == "dynamic":
            torch._dynamo.mark_dynamic(a, 0)
        elif case == "deterministic":
            self.enterContext(config.patch({"deterministic": True}))
        elif case == "native_matmul_off":
            self.enterContext(helion_config.patch(native_matmul=False))
        # Declined when scheduling, once lowered to a dot: Triton's native matmul
        # codegen emits it.
        elif case == "gathered_operand":
            fn = lambda a, i, b: a[i] @ b  # noqa: E731
            i = torch.randint(0, 80, (32,), device=DEVICE)
            args = [randn(80, 64), i, b]
            lowering = "tl.dot("
        elif case == "cpu_scalar":
            # The CPU scalar epilogue fuses into the matmul, so the kernel is Triton.
            fn = lambda a, b, c: a @ b + c  # noqa: E731
            args = [a, b, torch.tensor(2.0)]
            lowering = "tl.dot("
        expect_helion = case == "none"
        sources = self.compile(
            fn, *args, expected=fn(*args), tol=2e-2, expect_helion=expect_helion
        )
        self.assertEqual(
            len([s for s in sources if DOT_RE.search(s)]), int(expect_helion)
        )
        if not expect_helion:
            self.assertIn(lowering, self.code)

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
        # Helion's CuTe backend compiles no lgamma, so the kernel stays Triton.
        x = torch.rand(64, 100, device=DEVICE) + 0.5
        sources = self.compile(
            torch.lgamma, x, tol=1e-4, backend="cute", expect_helion=False
        )
        self.assertEqual(sources, [])
        (source,) = self.compile(torch.exp, x, tol=1e-4, backend="cute")
        self.assertIn('backend="cute"', source)


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

    def test_keeps_cuda_backend(self):
        scheduling = common.get_scheduling_for_device("cuda")
        self.assertIsInstance(scheduling(None), helion.inductor.HelionCUDAScheduling)
        with config.patch(cuda_backend="halide"):
            self.assertNotIsInstance(
                scheduling(None), helion.inductor.HelionCUDAScheduling
            )

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
        helion.inductor.disable()
        previous = common.custom_backend_passes.get("cuda")
        common.custom_backend_passes["cuda"] = custom_pass = object()
        try:
            helion.inductor.enable()
            self.assertIs(common.custom_backend_passes["cuda"], custom_pass)
        finally:
            helion.inductor.disable()
            common.custom_backend_passes["cuda"] = previous

    def test_disable(self):
        helion.inductor.disable()
        x = torch.randn(64, 100, device=DEVICE)
        self.assertEqual(self.compile(lambda x: x.sum(-1), x, expect_helion=False), [])

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
        # enable() registers helion_config with Inductor, whose FX graph cache key
        # includes its private _source_hash: another helion.inductor misses.
        x = torch.randn(64, 100, device=DEVICE)
        counters.clear()
        with fresh_cache():
            for source_hash in ("a", "a", "b"):
                torch._dynamo.reset()
                with helion_config.patch(_source_hash=source_hash):
                    torch.compile(lambda x: x.sum(-1))(x)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

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

    @parametrize(
        "case",
        [
            "none",
            "argmax",
            "dynamic",
            "strict_reduction",
            "strict_pointwise",
            "batch_invariant",
            "deterministic",
            "pytorch_libdevice",
        ],
    )
    def test_reduction_fallback(self, case):
        x = torch.randn(64, 100, device=DEVICE)
        fn = (lambda x: x.argmax(-1)) if case == "argmax" else (lambda x: x.sum(-1))
        if case == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        # "none" checks that x.sum(-1) is a Helion kernel without these modes.
        modes = {
            "strict_reduction": {"numerics": "strict_reduction"},
            "strict_pointwise": {"numerics": "strict_pointwise"},
            "batch_invariant": {"batch_invariant": True},
            "deterministic": {"deterministic": True},
            "pytorch_libdevice": {"eager_numerics.use_pytorch_libdevice": True},
        }
        expect_helion = case == "none"
        with config.patch(modes.get(case, {})):
            sources = self.compile(fn, x, tol=1e-4, expect_helion=expect_helion)
        self.assertEqual(len(sources), int(expect_helion))

    @helion_config.patch(fast_math=True)
    def test_fast_math(self):
        fn = torch.nn.functional.gelu
        (source,) = self.compile(fn, torch.randn(64, 100, device=DEVICE), tol=1e-4)
        self.assertIn("fast_math=True", source)


if __name__ == "__main__":
    unittest.main()
