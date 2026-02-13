import contextlib
import unittest.mock as mock

import torch
import torch._inductor.utils as inductor_utils
from torch._inductor import config
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import run_and_get_code


class TestTritonTemplateFusion(TestCase):
    """Small-scale tests for Triton template epilogue/prologue fusion.

    Patches is_big_gpu → True so tests run on GPUs with < 68 SMs.
    Uses max_mm_configs=1 to keep autotuning fast.
    autotune_fallback_to_aten=False ensures failure if no Triton choice is found.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            mock.patch.object(inductor_utils, "is_big_gpu", return_value=True)
        )
        cls._stack.enter_context(
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "autotune_fallback_to_aten": False,
                    "benchmark_epilogue_fusion": False,
                    "test_configs.max_mm_configs": 1,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def test_mm_relu_epilogue(self):
        """relu epilogue should be fused: expect 1 kernel launch, not 2."""
        def fn(a, b):
            return torch.relu(torch.mm(a, b))

        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        result, code = run_and_get_code(torch.compile(fn), a, b)
        torch.testing.assert_close(result, fn(a, b))
        # Epilogue fused → single kernel launch in generated wrapper
        # (kernel_launch_count is the number of .run( calls in the wrapper)
        kernel_launches = code[0].count(".run(")
        self.assertEqual(kernel_launches, 1, msg=f"Expected 1 kernel (epilogue fused), got {kernel_launches}")

    def test_mm_bias_epilogue(self):
        """bias-add epilogue should be fused: expect 1 kernel launch."""
        def fn(a, b, bias):
            return torch.mm(a, b) + bias

        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        bias = torch.randn(64, 64, device="cuda")
        result, code = run_and_get_code(torch.compile(fn), a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias))
        kernel_launches = code[0].count(".run(")
        self.assertEqual(kernel_launches, 1, msg=f"Expected 1 kernel (epilogue fused), got {kernel_launches}")

    def test_mm_cast_prologue(self):
        """fp16→fp32 cast prologue should be fused: expect 1 kernel launch."""
        def fn(a, b):
            return torch.mm(a.to(torch.float32), b.to(torch.float32))

        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        with config.patch({"prologue_fusion": True}):
            result, code = run_and_get_code(torch.compile(fn), a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-3, rtol=1e-3)
        kernel_launches = code[0].count(".run(")
        self.assertEqual(kernel_launches, 1, msg=f"Expected 1 kernel (prologue fused), got {kernel_launches}")

    def test_mm_relu_epilogue_correctness(self):
        """Basic correctness check across multiple sizes."""
        def fn(a, b):
            return torch.relu(torch.mm(a, b))

        for M, K, N in [(32, 32, 32), (128, 64, 128)]:
            a = torch.randn(M, K, device="cuda")
            b = torch.randn(K, N, device="cuda")
            torch.testing.assert_close(torch.compile(fn)(a, b), fn(a, b))


if __name__ == "__main__":
    run_tests()
