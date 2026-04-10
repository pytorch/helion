from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from packaging import version
import pytest
import torch
import torch.nn.functional as F

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import check_example
from helion._testing import get_nvidia_gpu_model
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfA10G
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfCudaSharedMemoryLessThan
from helion._testing import skipIfFn
from helion._testing import skipIfPallas
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import xfailIfCute
from helion._testing import xfailIfPallas
from helion.runtime.config import Config
from helion.runtime.ref_mode import is_ref_mode_enabled

_orig_matmul_fp32_precision: str = "none"
_orig_cudnn_fp32_precision: str = "none"


def _compile_only(
    fn: helion.Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> object:
    bound = fn.bind(args)
    if kwargs:
        config = Config(
            # pyrefly: ignore [bad-argument-type]
            **kwargs
        )
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = bound.config_spec.default_config()
    for key in bound.config_spec.unsupported_config_keys(config.config):
        config.config.pop(key, None)
    if is_ref_mode_enabled(bound.kernel.settings):
        bound._config = config
        return bound
    return bound.compile_config(config)


def setUpModule() -> None:
    global _orig_matmul_fp32_precision, _orig_cudnn_fp32_precision
    _orig_matmul_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    _orig_cudnn_fp32_precision = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def tearDownModule() -> None:
    torch.backends.cuda.matmul.fp32_precision = _orig_matmul_fp32_precision
    torch.backends.cudnn.conv.fp32_precision = _orig_cudnn_fp32_precision


@onlyBackends(["triton", "cute", "pallas"])
class TestExamples(RefEagerTestBase, TestCase):
    def test_add(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example("add", args, sum(args), block_sizes=[128, 1], flatten_loop=True)

    def test_add_loop_order(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "add", args, sum(args), block_sizes=[256, 128], loop_orders=[[1, 0]]
        )

    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    def test_matmul(self):
        args = (
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 512], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul",
            args,
            args[0] @ args[1],
            block_sizes=[128, 128, 128],
        )

    @xfailIfCute("CuTe barrier-based split-K example is still unsupported")
    @xfailIfPallas("missing barrier implementation")
    @skipIfTileIR("PassManager::run failed")
    @skipIfXPU("Split-K barrier not supported on XPU backend")
    def test_split_k_barrier(self):
        m, k, n = 64, 512, 64
        a = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        expected = a @ b

        check_example(
            "split_k_barrier",
            (a, b),
            expected,
            fn_name="split_k_matmul",
            block_sizes=[16, 8, 16, 16, 16],
            pid_type="persistent_blocked",
            split_k=64,
        )

    @xfailIfCute("CuTe barrier-based split-K example is still unsupported")
    @xfailIfPallas("missing barrier implementation")
    @skipIfTileIR("PassManager::run failed")
    @skipIfRefEager("Test requires compiled kernel with specific config")
    def test_split_k_barrier_accuracy(self):
        """Test split_k_barrier with a shape that exposes accuracy issues.

        This test uses shape (64, 33, 64) where K is not divisible by split_k.
        The bug manifests after multiple kernel executions - errors accumulate
        due to improper handling of the tmp tensor across invocations.
        """
        from examples.split_k_barrier import split_k_matmul

        m, k, n = 64, 33, 64
        config = helion.Config(
            block_sizes=[16, 8, 16, 16, 16],
            pid_type="persistent_blocked",
            split_k=32,
        )

        # Compile once and reuse - this triggers the accumulating error bug
        torch.manual_seed(0)
        a0 = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
        b0 = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        bound = split_k_matmul.bind((a0, b0))
        compiled = bound.compile_config(config)

        # Run multiple iterations - errors accumulate starting around iteration 2-3
        for seed in range(5):
            torch.manual_seed(seed)
            a = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
            b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
            expected = a @ b
            result = compiled(a, b)

            torch.testing.assert_close(
                result,
                expected,
                atol=1e-1,
                rtol=1e-2,
                msg=f"Accuracy failure at iteration {seed}",
            )

    def test_matmul_bwd(self):
        """Test backward pass for matmul via matmul_autograd."""
        mod = import_path(EXAMPLES_DIR / "matmul.py")
        # Set a fixed config to avoid autotuning in CI
        config = helion.Config(block_sizes=[16, 16, 16])
        mod.matmul.configs = [config]

        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )

        mat1_ref = mat1.detach().clone().requires_grad_(True)
        mat2_ref = mat2.detach().clone().requires_grad_(True)
        ref_out = torch.matmul(mat1_ref, mat2_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        result = mod.matmul_autograd(mat1, mat2)
        result.backward(grad_out)

        torch.testing.assert_close(result, ref_out, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat1.grad, mat1_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat2.grad, mat2_ref.grad, atol=1e-1, rtol=1e-2)

    def test_addmm_bwd(self):
        """Test backward pass for addmm via addmm_autograd."""
        mod = import_path(EXAMPLES_DIR / "matmul.py")
        # Set a fixed config to avoid autotuning in CI
        config = helion.Config(block_sizes=[16, 16, 16])
        mod.matmul.configs = [config]

        bias = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        alpha, beta = 2.0, 0.5

        bias_ref = bias.detach().clone().requires_grad_(True)
        mat1_ref = mat1.detach().clone().requires_grad_(True)
        mat2_ref = mat2.detach().clone().requires_grad_(True)
        ref_out = torch.addmm(bias_ref, mat1_ref, mat2_ref, alpha=alpha, beta=beta)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        result = mod.addmm_autograd(bias, mat1, mat2, alpha, beta)
        result.backward(grad_out)

        torch.testing.assert_close(result, ref_out, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat1.grad, mat1_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat2.grad, mat2_ref.grad, atol=1e-1, rtol=1e-2)

    @xfailIfCute("CuTe matmul+layernorm example still fails codegen/runtime")
    def test_matmul_layernorm_static_shapes(self):
        args = (
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul_layernorm",
            args,
            torch.nn.functional.layer_norm(
                (args[0] @ args[1]),
                normalized_shape=(512,),
                weight=args[2],
                bias=args[3],
            ),
            block_sizes=[16, 16],
            static_shapes=True,
        )

    @xfailIfCute("CuTe matmul+layernorm example still fails codegen/runtime")
    @xfailIfPallas("JAX tracer error with dynamic shapes")
    def test_matmul_layernorm_dynamic_shapes(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul_layernorm",
            args,
            torch.nn.functional.layer_norm(
                (args[0] @ args[1]),
                normalized_shape=(400,),
                weight=args[2],
                bias=args[3],
            ),
            block_sizes=[16, 16],
            static_shapes=False,
        )

    @unittest.skipIf(
        version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"),
        "Requires torch 2.8+",
    )
    def test_bmm(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([16, 768, 1024], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "bmm",
            args,
            torch.bmm(args[0], args[1]),
            block_sizes=[16, 16, 16, 16],
        )

    @skipIfFn(
        lambda: _get_backend() == "cute", "CuTe FP8 GEMM example is not supported yet"
    )
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_fp8_gemm(self):
        # Create FP32 tensors and convert to FP8
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)

        # Convert to FP8 format
        x_fp8 = x.to(torch.float8_e4m3fn)
        y_fp8 = y.to(torch.float8_e4m3fn)

        args = (x_fp8, y_fp8)

        # Import the reference implementation
        mod = import_path(EXAMPLES_DIR / "fp8_gemm.py")
        expected = mod.reference_fp8_gemm_pytorch(x_fp8, y_fp8)

        check_example(
            "fp8_gemm",
            args,
            expected,
            block_sizes=[16, 16, 32],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfCute("CuTe template closure example still exceeds runtime resources")
    def test_template_via_closure0(self):
        bias = torch.randn([1, 512], device=DEVICE, dtype=HALF_DTYPE)
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1] + bias),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="pointer",
            l2_grouping=64,
        )

    @xfailIfCute("CuTe template closure example still exceeds runtime resources")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("Failed on XPU - https://github.com/pytorch/helion/issues/795")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_template_via_closure1(self):
        bias = torch.randn([1, 512], device=DEVICE, dtype=HALF_DTYPE)
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1] + bias),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="block_ptr",
            l2_grouping=64,
        )

    @xfailIfCute("CuTe template closure example still exceeds runtime resources")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_template_via_closure2(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda x, _: torch.nn.functional.relu(x),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1]),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="block_ptr",
            l2_grouping=64,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_looped(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
            reduction_loop=32,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_decomposed(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_decomposed",
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
        )

    def test_softmax_two_pass(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_two_pass",
            emit_code=False,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_two_pass_block_ptr(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_two_pass",
            emit_code=False,
            block_sizes=[8, 64],
            indexing="block_ptr",
        )

    @xfailIfPallas("BlockSpec tiling failure")
    def test_cross_entropy(self):
        n, v = 128, 1000
        args = (
            torch.randn(n, v, device=DEVICE, dtype=torch.float32),
            torch.randint(0, v, (n,), device=DEVICE, dtype=torch.long),
        )
        check_example(
            "cross_entropy",
            args,
            torch.nn.functional.cross_entropy(*args),
        )

    @xfailIfCute("CuTe Welford example still returns incorrect results")
    def test_welford(self):
        s, d = 128, 1024
        weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

        check_example(
            "welford",
            (weight, bias, x),
            torch.nn.functional.layer_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=weight,
                bias=bias,
                eps=1e-05,
            ),
        )

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe low-memory dropout example is flaky and currently skipped",
    )
    def test_low_mem_dropout(self):
        from examples.low_mem_dropout import low_mem_dropout
        from examples.low_mem_dropout import low_mem_dropout_bwd

        p = 0.25
        size = 1024
        block_size = 512
        seed = 123
        seed2 = 456
        x = torch.randn(size=(size,)).to(device=DEVICE)
        out_fwd = _compile_only(
            low_mem_dropout, (p, x, seed), block_sizes=[block_size]
        )(p, x, seed)

        grad_y = torch.ones_like(x)
        bwd = _compile_only(
            low_mem_dropout_bwd,
            (p, grad_y, seed),
            block_sizes=[block_size],
        )
        grad_x = bwd(p, grad_y, seed)

        grad_x2 = bwd(p, grad_y, seed2)

        mask_fwd = out_fwd != 0
        mask_bwd = grad_x != 0
        self.assertTrue(
            torch.equal(mask_fwd, mask_bwd),
            "Same elements should be dropped in fwd and bwd with the same seed",
        )

        mask_bwd2 = grad_x2 != 0
        self.assertFalse(
            torch.equal(mask_bwd, mask_bwd2),
            "Different elements should be dropped when using a different seed",
        )

        check_example(
            "low_mem_dropout",
            (p, grad_y, seed),
            grad_x,
            block_sizes=[block_size],
        )

    @xfailIfCute("CuTe bf16 x int16 example still returns incorrect results")
    @xfailIfPallas("precision differences with bf16xint16 operations on pallas")
    @skipIfTileIR("precision differences with bf16xint16 operations on tileir")
    @skipIfRocm("precision differences with bf16xint16 operations on rocm")
    @skipIfXPU("precision differences with bf16xint16 operations on xpu")
    def test_bf16xint16(self):
        from examples.bf16xint16_gemm import reference_bf16xint16_pytorch

        m, k, n = 65536, 1024, 1280

        x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)

        check_example(
            "bf16xint16_gemm",
            (x, w),
            reference_bf16xint16_pytorch(x, w, False),
            fn_name="_bf16xint16_gemm",
        )

        x_int16 = torch.randint(
            -(2**15), 2**15 - 1, (m, k), device=DEVICE, dtype=torch.int16
        )
        w_bf16 = torch.randn([k, n], device=DEVICE, dtype=torch.bfloat16)

        check_example(
            "bf16xint16_gemm",
            (x_int16, w_bf16),
            reference_bf16xint16_pytorch(x_int16, w_bf16, True),
            fn_name="_int16xbf16_gemm",
        )

    def test_rms_norm_fwd(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256], device=DEVICE, dtype=HALF_DTYPE),
            1e-5,
        )
        # Import and use the reference implementation from rms_norm.py
        mod = import_path(EXAMPLES_DIR / "rms_norm.py")
        expected = mod.rms_norm_pytorch(*args)

        check_example(
            "rms_norm",
            args,
            (expected, None),  # Expected: (output, 1/rms)
            fn_name="rms_norm_fwd",
            block_sizes=[16],
            indexing="pointer",
        )

    def test_swiglu_bwd(self):
        """Test backward pass for swiglu."""
        x1, x2 = [
            torch.randn(1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)
        ]

        out = F.silu(x1) * x2

        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        args = (
            grad_out,
            x1,
            x2,
        )

        check_example(
            "swiglu",
            args,
            (x1.grad, x2.grad),
            fn_name="swiglu_bwd",
        )

    @xfailIfCute("CuTe RMSNorm backward example still returns incorrect results")
    @skipIfPallas("Generated Pallas code causes Mosaic internal SegFault")
    def test_rms_norm_bwd(self):
        """Test backward pass for rms norm weight gradient."""
        batch_size, dim = 32, 64
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
        grad_out = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        eps = 1e-5

        # Compute forward pass to get rms
        from examples.rms_norm import rms_norm_fwd

        # Create configured kernel with explicit config
        config = helion.Config(block_size=32, num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(rms_norm_fwd.fn, config=config)
        y, rms = configured_kernel(x, weight, eps)

        # Compute expected gradients with PyTorch
        x_torch = x.detach().clone().requires_grad_(True)
        weight_torch = weight.detach().clone().requires_grad_(True)
        y_torch = torch.nn.functional.rms_norm(x_torch, [dim], weight_torch, eps)
        y_torch.backward(grad_out)

        # Test the kernel using check_example
        args = (
            grad_out,
            x,
            weight,
            rms,
        )

        # rms_norm_bwd_dw returns grad_weight
        check_example(
            "rms_norm",
            args,
            (x_torch.grad, weight_torch.grad),  # Expected: grad_weight
            fn_name="rms_norm_bwd",
            block_size=[32, 1],
            num_warps=4,
            num_stages=3,
            rtol=1e-2,
            atol=1e-2,
        )

    @xfailIfPallas("BlockSpec tiling failure")
    def test_embedding_pointers(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "embedding",
            args,
            torch.nn.functional.embedding(*args),
            block_sizes=[1, 256],
            indexing="pointer",
        )

    @xfailIfPallas("BlockSpec tiling failure")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_embedding_block_ptr(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "embedding",
            args,
            torch.nn.functional.embedding(*args),
            block_sizes=[8, 64],
            indexing="block_ptr",
            pid_type="xyz",
        )

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe attention pointer destabilizes later cute tests when it fails in-process",
    )
    def test_attention_pointer(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            block_sizes=[1, 64, 32],
            indexing="pointer",
        )

    @xfailIfCute(
        "CuTe attention block-pointer example still exceeds thread-block limits"
    )
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_attention_block_pointer(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            block_sizes=[16, 32, 16],
            num_stages=1,
            indexing="block_ptr",
        )

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe dynamic attention destabilizes later cute tests when it fails in-process",
    )
    @xfailIfPallas("JAX tracer error with dynamic shapes")
    def test_attention_dynamic(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            fn_name="attention_dynamic",
            block_sizes=[1, 64, 32],
        )

    @xfailIfPallas("BlockSpec tiling failure")
    def test_concat(self):
        args = (
            torch.randn(512, 500, device=DEVICE),
            torch.randn(512, 512, device=DEVICE),
        )
        check_example(
            "concatenate",
            args,
            torch.cat(args, dim=1),
            fn_name="concat2d_dim1",
        )

    @xfailIfPallas("BlockSpec tiling failure")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_concat_block_ptr(self):
        args = (
            torch.randn(222, 100, device=DEVICE),
            torch.randn(222, 151, device=DEVICE),
        )
        check_example(
            "concatenate",
            args,
            torch.cat(args, dim=1),
            fn_name="concat2d_dim1",
            indexing="block_ptr",
            block_sizes=[128, 64],
        )

    @xfailIfPallas("BlockSpec tiling failure")
    def test_jagged_dense_add(self):
        mod = import_path(EXAMPLES_DIR / "jagged_dense_add.py")
        args = (
            *mod.random_jagged_2d(500, 5000, device=DEVICE),
            torch.randn(500, 5000, device=DEVICE),
        )
        check_example(
            "jagged_dense_add",
            args,
            mod.jagged_dense_add_2d_reference(*args),
            fn_name="jagged_dense_add_2d",
        )

    @xfailIfCute("CuTe jagged dense bmm example still returns incorrect results")
    @xfailIfPallas("tensor-derived if-predicates not supported")
    @skipIfXPU("Jagged tensor operations not fully supported on XPU")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_dense_bmm(self):
        mod = import_path(EXAMPLES_DIR / "jagged_dense_bmm.py")
        seq_offsets, jagged, dense, bias = mod.random_input(
            D=32, K=24, batch_size=16, max_seq_len=32, dtype=torch.float32
        )
        args = (seq_offsets, jagged, dense, bias)
        check_example(
            "jagged_dense_bmm",
            args,
            mod.jagged_dense_bmm_reference(*args),
        )

    @xfailIfPallas("tensor-derived if-predicates not supported")
    @skipIfRefEager("Test has skip_accuracy=True and doesn't call assert_close")
    def test_moe_matmul_ogs(self):
        mod = import_path(EXAMPLES_DIR / "moe_matmul_ogs.py")

        B = 1000  # tokens / rows
        K = 500  # hidden size
        N = 200  # output size
        n_experts = 30
        A = torch.randn(B, K, device=DEVICE, dtype=HALF_DTYPE)
        W = torch.randn(n_experts, K, N, device=DEVICE, dtype=HALF_DTYPE)
        top1_expert_per_token = torch.randint(n_experts, (B,), device=DEVICE)

        args = (A, W, top1_expert_per_token)
        helion_kernel_args = mod.moe_matmul_ogs_helion_kernel_args_gen(
            A, W, top1_expert_per_token
        )
        check_example(
            "moe_matmul_ogs",
            helion_kernel_args,
            mod.moe_matmul_ogs_reference(*args),
            block_sizes=[16, 16, 16],
            skip_accuracy=True,  # TODO(yf225): fix unstable numerics
        )

    @xfailIfPallas("InductorLoweringError")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_matmul_split_k(self):
        args = (
            torch.randn(64, 1024, device=DEVICE),
            torch.randn(1024, 64, device=DEVICE),
        )
        check_example(
            "matmul_split_k",
            args,
            torch.matmul(*args),
            indexing="block_ptr",
            block_sizes=[16, 16, 32],
            split_k=8,
        )

    def test_sum(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "sum",
            args,
            torch.sum(args[0], dim=-1),
            fn_name="sum_kernel",
            block_sizes=[8],
        )

    def test_long_sum_manual(self):
        # longsum_manual uses hl.register_block_size to get a static bound for the
        # inner reduction loop, so range() receives a plain Python int — no JAX
        # tracer wrapping.  Use n=65536 (2x the 32768 block size) to exercise two
        # reduction loop iterations on Pallas.
        x = torch.randn([4, 65536], device=DEVICE, dtype=torch.float32)
        check_example(
            "long_sum",
            (x,),
            x.sum(-1),
            fn_name="longsum_manual",
        )

    @xfailIfCute("CuTe jagged mean example still fails lowering/runtime")
    @xfailIfPallas("JAX tracer error with dynamic shapes")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_mean(self):
        num_rows, max_cols = 32, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        feature_counts = torch.randint(
            1, M + 1, (num_rows,), dtype=torch.int32, device=DEVICE
        )
        args = (x_data, x_offsets, feature_counts, M)

        mod = import_path(EXAMPLES_DIR / "jagged_mean.py")
        expected = mod.reference_jagged_mean_kernel_pytorch(
            x_data, x_offsets, feature_counts, M
        )

        check_example(
            "jagged_mean",
            args,
            expected,
            fn_name="jagged_mean_kernel",
            block_sizes=[16, 8, 16],
        )

    @xfailIfCute("CuTe segment reduction example is not supported yet")
    @xfailIfPallas("requires triton module")
    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_segment_reduction(self):
        num_nodes = 100
        num_edges = 1000
        num_features = 32
        dtype = torch.float32

        # Create sorted indices for segmented reduction
        indices = torch.randint(0, num_nodes, (num_edges,), device=DEVICE).sort()[0]
        input_data = torch.randn(num_edges, num_features, device=DEVICE, dtype=dtype)

        args = (indices, input_data, num_nodes)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "segment_reduction.py")
        expected = mod.segmented_reduction_pytorch(*args)

        check_example(
            "segment_reduction",
            args,
            expected,
            fn_name="segmented_reduction_helion",
        )

    @xfailIfCute("CuTe persistent attention example still exceeds thread-block limits")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_attention_persistent_interleaved_l2_grouping(self):
        """Test attention with persistent interleaved execution and L2 grouping for optimal performance."""
        args = (
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )

        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            block_sizes=[16, 32, 16],
            num_stages=1,
            pid_type="persistent_interleaved",
            l2_grouping=4,
            indexing="block_ptr",
        )

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe FP8 attention destabilizes later cute tests when it fails in-process",
    )
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_fp8_attention(self):
        batch = 2
        heads = 4
        seq_len = 256
        head_dim = 64

        # Create FP16 tensors
        q = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )

        # Import the module
        mod = import_path(EXAMPLES_DIR / "fp8_attention.py")

        # Prepare FP8 inputs using the module's preprocessing function
        q_fp8, k_fp8, v_fp8 = mod.preprocess_fp8_attention_inputs(q, k, v)
        args = (q_fp8, k_fp8, v_fp8, batch, heads)

        # Get expected output from kernel
        expected = mod.fp8_attention_pytorch(q, k, v)()

        check_example(
            "fp8_attention",
            args,
            expected,
            fn_name="fp8_attention_kernel",
            block_sizes=[64, 64],
            atol=0.2,
            rtol=0.1,
        )

    def test_layernorm_with_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, bias)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),  # Expected: (output, mean, rstd)
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    def test_layernorm_no_bias(self):
        """Test forward pass for layer normalization without bias."""
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, None)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),  # Expected: (output, mean, rstd)
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallas(
        "Out-of-bounds slice when reduction_loops doesn't evenly divide the "
        "reduction dimension (e.g. reduction_loops=32 on dim=48 generates "
        "pl.ds(32, 32) which exceeds bounds)"
    )
    def test_layernorm_reduction_not_divisible(self):
        """Reduction loop OOB when reduction_loops doesn't divide the reduction dim."""
        batch_size = 4
        dim = 48  # not divisible by reduction_loops=32
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)

        args = (x, [dim], weight, bias, 1e-5)
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),
            fn_name="layer_norm_fwd",
            block_size=1,
            reduction_loops=32,
        )

    @xfailIfCute("CuTe LayerNorm backward example still returns incorrect results")
    @xfailIfPallas("InductorLoweringError")
    @skipIfA10G("accuracy check fails on A10G GPUs")
    def test_layernorm_bwd(self):
        """Test combined backward pass for layer norm with bias, including regression coverage."""

        cases = (
            {
                "batch_size": 32,
                "dim": 64,
            },
            {
                "batch_size": 1152 * 1000,
                "dim": 16,
            },
        )

        eps = 1e-4
        atol = 3e-2
        rtol = 5e-2

        for idx, case in enumerate(cases):
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(idx)

            batch_size = case["batch_size"]
            dim = case["dim"]

            x = -2.3 + 0.5 * torch.randn(
                [batch_size, dim], device=DEVICE, dtype=HALF_DTYPE
            )
            weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
            bias = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
            grad_out = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)

            # Compute mean, var, and rstd in fp32 to match Helion forward kernel output
            x_fp32 = x.to(torch.float32)
            mean = x_fp32.mean(dim=-1)
            var = x_fp32.var(dim=-1, unbiased=False)
            rstd = torch.rsqrt(var + eps)

            x_ref = x.clone().detach().requires_grad_(True)
            weight_ref = weight.clone().detach().requires_grad_(True)
            bias_ref = bias.clone().detach().requires_grad_(True)

            y_ref = torch.nn.functional.layer_norm(
                x_ref, [dim], weight_ref, bias_ref, eps
            )
            y_ref.backward(grad_out.detach())

            expected = (
                x_ref.grad.detach(),
                weight_ref.grad.detach(),
                bias_ref.grad.detach(),
            )

            args = (grad_out, x, mean, rstd, weight, True)

            check_example(
                "layer_norm",
                args,
                expected,
                fn_name="layer_norm_bwd",
                block_sizes=[32, 1],
                num_warps=4,
                num_stages=3,
                rtol=rtol,
                atol=atol,
            )

    def test_softmax_bwd(self):
        m, n = 2048, 2048
        x = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16)

        from examples.softmax import softmax_two_pass

        config = helion.Config(block_size=[128, 128], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(softmax_two_pass.fn, config=config)
        y = configured_kernel(x)

        x_torch = x.detach().clone().requires_grad_(True)
        y_torch = torch.nn.functional.softmax(x_torch, dim=-1)
        y_torch.backward(grad_out)

        check_example(
            "softmax",
            (grad_out, y),
            x_torch.grad,
            fn_name="softmax_bwd",
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm_without_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, None)
        # Test returns (output, mean, rstd) tuple
        expected_out = torch.nn.functional.layer_norm(x, [64], weight)
        expected = (expected_out, None, None)
        check_example(
            "layer_norm",
            args,
            expected,
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    @xfailIfCute("CuTe jagged softmax example still fails lowering/runtime")
    @xfailIfPallas("JAX tracer error with dynamic shapes")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_softmax(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        args = (x_data, x_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_softmax.py")
        expected = mod.reference_jagged_softmax_pytorch(x_data, x_offsets)

        check_example(
            "jagged_softmax",
            args,
            expected,
            fn_name="jagged_softmax_kernel",
            block_sizes=[16, 8, 16, 16],
        )

    @xfailIfCute("CuTe jagged HSTU attention example is not supported yet")
    @xfailIfPallas("tensor-derived if-predicates not supported")
    @skipIfXPU("Jagged tensor operations not fully supported on XPU")
    def test_jagged_hstu_attn(self):
        batch_size = 4
        max_seq_len = 64
        heads = 8
        head_dim = 32

        # Generate random sequence lengths
        min_seq_len = max_seq_len // 2
        seq_lengths = torch.randint(
            min_seq_len,
            max_seq_len + 1,
            (batch_size,),
            dtype=torch.int32,
            device=DEVICE,
        )
        seq_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=DEVICE),
                torch.cumsum(seq_lengths, dim=0),
            ]
        )
        total_seq_len = int(seq_offsets[-1].item())

        # Create input tensors: [total_seq_len, heads, head_dim]
        q = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        k = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        v = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # The kernel expects: max_seq_len, alpha, q, k, v, seq_offsets
        alpha = 1.0 / v.size(2) ** 2
        args = (max_seq_len, alpha, q, k, v, seq_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_hstu_attn.py")
        expected = mod.reference_jagged_hstu_kernel_pytorch(
            q, k, v, seq_offsets, None, max_seq_len
        )

        # Patch to use core silu decomposition instead of inductor's custom decomposition from pytorch PR #171723.
        # This ensures consistent codegen both torch 2.9 (stable) and nightly versions.
        from torch._decomp.decompositions import silu
        import torch._inductor.decomposition as inductor_decomp

        # Clear cache since fast_random_decomps() caches a copy of decompositions
        if hasattr(inductor_decomp.fast_random_decomps, "cache_clear"):
            inductor_decomp.fast_random_decomps.cache_clear()
        with patch.dict(
            inductor_decomp.decompositions, {torch.ops.aten.silu.default: silu}
        ):
            check_example(
                "jagged_hstu_attn",
                args,
                expected,
                fn_name="_helion_jagged_attention_kernel",
                block_sizes=[16, 16],
                atol=1e-2,
                rtol=1e-2,
            )

    @xfailIfPallas("tensor-derived if-predicates not supported")
    def test_grouped_gemm_jagged(self):
        # Build small jagged grouped GEMM inputs
        torch.manual_seed(0)
        G = 3
        K, N = 64, 64
        dtype = torch.bfloat16
        group_A = [
            torch.randn(32 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous()
            for i in range(G)
        ]
        B_shared = torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()

        # Pack A and offsets
        M_sizes = [int(a.size(0)) for a in group_A]
        starts = [0]
        for m in M_sizes:
            starts.append(starts[-1] + m)
        group_offsets = torch.tensor(starts, device=DEVICE, dtype=torch.int32)
        A_packed = torch.cat(group_A, dim=0).contiguous()

        # Reference result
        expected = torch.cat([a @ B_shared for a in group_A], dim=0)

        # Run kernel and check
        args = (A_packed, B_shared, group_offsets)
        check_example(
            "grouped_gemm",
            args,
            expected,
            fn_name="grouped_gemm_jagged",
        )

    @xfailIfCute(
        "CuTe persistent grouped jagged GEMM example still fails lowering/runtime"
    )
    @xfailIfPallas("CUDA-specific code paths")
    def test_grouped_gemm_jagged_persistent(self):
        # Build small jagged grouped GEMM inputs
        torch.manual_seed(0)
        G = 3
        K, N = 64, 64
        dtype = torch.bfloat16
        group_A = [
            torch.randn(32 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous()
            for i in range(G)
        ]
        B_shared = torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()

        # Pack A and offsets
        M_sizes = [int(a.size(0)) for a in group_A]
        starts = [0]
        for m in M_sizes:
            starts.append(starts[-1] + m)
        group_offsets = torch.tensor(starts, device=DEVICE, dtype=torch.int32)
        A_packed = torch.cat(group_A, dim=0).contiguous()

        # Reference result
        expected = torch.cat([a @ B_shared for a in group_A], dim=0)

        # Run kernel and check
        args = (
            A_packed,
            B_shared,
            group_offsets,
        )
        check_example(
            "grouped_gemm",
            args,
            expected,
            fn_name="grouped_gemm_jagged_persistent",
        )

    def test_geglu(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
        )
        check_example(
            "geglu",
            args,
            torch.nn.functional.gelu(args[0], approximate="tanh") * args[1],
            block_sizes=[1024],
            num_warps=4,
            num_stages=3,
        )

    def test_geglu_bwd(self):
        x1, x2 = [
            torch.randn(1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)
        ]

        out = torch.nn.functional.gelu(x1, approximate="tanh") * x2
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        args = (grad_out, x1, x2)

        check_example(
            "geglu",
            args,
            (x1.grad, x2.grad),
            fn_name="geglu_bwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    def test_swiglu(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
        )
        check_example(
            "swiglu",
            args,
            torch.nn.functional.silu(args[0]) * args[1],
            fn_name="swiglu_fwd",
            block_sizes=[1024],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallas("InductorLoweringError")
    def test_jsd(self):
        args = (
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            None,
        )

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jsd.py")
        expected = mod.TorchJSDBaseline()
        check_example(
            "jsd",
            args,
            (expected(*args), None),
            fn_name="jsd_forward",
            emit_code=False,
            block_sizes=[1, 4096],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallas("operation not supported on TPU")
    def test_kl_div(self):
        if _get_backend() == "cute" and "B200" in get_nvidia_gpu_model():
            pytest.xfail("CuTe KL-div example still launches out of resources on B200")
        args = (
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).softmax(
                dim=-1
            ),
        )
        torch_kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(
            device=DEVICE
        )
        check_example(
            "kl_div",
            args,
            torch_kl_div(*args),
            fn_name="kl_div_forward",
            emit_code=False,
            block_sizes=[1, 4096],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallas("BackendError on pallas")
    def test_gather_gemv(self):
        args = (
            torch.randn([4, 512, 512], device=DEVICE, dtype=torch.float32),
            torch.randint(0, 4, [2], device=DEVICE, dtype=torch.int32),
            torch.randn([512], device=DEVICE, dtype=torch.float32),
        )

        def expected(w, idx, x):
            return w[idx].to(x.dtype) @ x

        check_example(
            "gather_gemv",
            args,
            expected(*args),
            fn_name="gather_gemv",
            emit_code=False,
            block_sizes=[16, 16],
            num_warps=8,
            num_stages=1,
        )

    @xfailIfCute("CuTe int4 GEMM example is not supported yet")
    @xfailIfPallas("int4 unpacking not supported on pallas")
    def test_int4_gemm(self):
        # Matrix dimensions
        M, K, N = 256, 512, 256

        # Create bfloat16 matrix A
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        # Create packed int4 matrix B
        # Generate random int4 values in range [-8, 7]
        B_unpacked = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=DEVICE)

        # Pack two int4 values per int8
        B_reshaped = B_unpacked.reshape(K // 2, 2, N).permute(1, 0, 2)
        B_packed = ((B_reshaped[0] & 0xF) | (B_reshaped[1] << 4)).to(torch.int8)

        # Convert unpacked to bfloat16 for expected result
        B_unpacked_bf16 = B_unpacked.to(torch.bfloat16)
        expected = torch.matmul(A, B_unpacked_bf16)

        args = (A, B_packed)

        check_example(
            "int4_gemm",
            args,
            expected,
            fn_name="matmul_bf16_int4",
            block_sizes=[64, 64, 32],
            num_warps=4,
            num_stages=3,
            rtol=2e-1,
            atol=1.0,
        )

    @xfailIfCute("CuTe NVFP4 GEMM example is not supported yet")
    @xfailIfPallas("NVFP4 is NVIDIA-specific")
    def test_nvfp4_gemm(self):
        from examples.nvfp4_gemm import pack_fp4
        from examples.nvfp4_gemm import quantize_fp4_e2m1
        from examples.nvfp4_gemm import reference_nvfp4_matmul

        M, K, N = 256, 128, 256

        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)

        W_quantized = quantize_fp4_e2m1(W)
        W_packed = pack_fp4(W_quantized)

        args = (A, W_packed)
        expected = reference_nvfp4_matmul(A, W_packed)

        check_example(
            "nvfp4_gemm",
            args,
            expected,
            fn_name="nvfp4_matmul",
            block_sizes=[64, 64, 32],
            num_warps=4,
            num_stages=3,
            rtol=2e-1,
            atol=1.0,
        )

    @xfailIfCute("CuTe jagged sum example still returns incorrect results")
    @xfailIfPallas("JAX tracer error")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_sum(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        args = (x_data, x_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_sum.py")
        expected = mod.reference_jagged_sum_kernel_pytorch(x_data, x_offsets)

        check_example(
            "jagged_sum",
            args,
            expected,
            fn_name="jagged_sum_kernel",
            block_sizes=[16, 8, 16],
        )

    def test_fused_linear_jsd(self):
        beta = 0.5
        ignore_index = -100
        temperature = 1.0
        m, n, k = 64, 128, 256

        student_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        teacher_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        student_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        teacher_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        student_logits = student_input @ student_weight.T
        teacher_logits = teacher_input @ teacher_weight.T

        args = (
            beta,
            ignore_index,
            temperature,
            student_logits,
            teacher_logits,
        )

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")
        # fused_linear_jsd_pytorch signature:
        # (beta, ignore_index, temperature, student_weight, teacher_weight, student_input, teacher_input)
        expected = mod.fused_linear_jsd_pytorch(
            beta,
            ignore_index,
            temperature,
            student_weight,
            teacher_weight,
            student_input,
            teacher_input,
        )

        check_example(
            "fused_linear_jsd",
            args,
            expected,
            fn_name="fused_linear_jsd_kernel",
            block_sizes=[64],
        )

    @xfailIfCute("CuTe jagged layer norm example still fails lowering/runtime")
    @xfailIfPallas("JAX tracer error")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_layer_norm(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        eps = 1e-6
        args = (x_data, x_offsets, eps)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_layer_norm.py")
        expected = mod.reference_jagged_layer_norm_pytorch(x_data, x_offsets, eps)

        check_example(
            "jagged_layer_norm",
            args,
            expected,
            fn_name="jagged_layer_norm_kernel",
            block_sizes=[4, 8, 8, 8, 8, 8, 8],
        )

    def test_exp_fwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
        args = (x,)
        check_example(
            "exp",
            args,
            torch.exp(x),
            fn_name="exp_fwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    def test_exp_bwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16).requires_grad_(
            True
        )
        y = torch.exp(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        torch_out = x.grad
        args = (
            grad_out,
            y,
        )
        check_example(
            "exp",
            args,
            torch_out,
            fn_name="exp_bwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfCute(
        "CuTe squeeze-and-excitation forward still exceeds thread-block limits"
    )
    @skipIfRocm("Triton ROCm store-forwarding bug: stale global memory reads")
    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    @skipIfXPU("Squeeze-and-excitation network not supported on XPU")
    def test_squeeze_and_excitation_net_fwd(self):
        m, n, k = 128, 128, 128
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float32)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)

        args = (x, a, b)

        expected_out = torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b))
        c = torch.relu(x @ a)
        d = torch.sigmoid(c @ b)

        check_example(
            "squeeze_and_excitation_net",
            args,
            (expected_out, c, d),
            fn_name="squeeze_and_excitation_net_fwd",
            block_sizes=[128, 128, 128, 128],
            num_warps=4,
            num_stages=2,
            atol=0.15,
        )

    @xfailIfCute("CuTe squeeze-and-excitation backward still fails lowering/runtime")
    @xfailIfPallas("conflicting tiling patterns")
    @skipIfA10G("failure on a10g")
    @skipIfXPU("Squeeze-and-excitation network not supported on XPU")
    @skipIfTileIR("accuracy failure")
    def test_squeeze_and_excitation_net_bwd_dx(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, a, b, c, d)
        expected = x_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_dx",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.3,
        )

    @xfailIfCute("CuTe squeeze-and-excitation backward still fails lowering/runtime")
    @xfailIfPallas("tensor accessed with conflicting tiling patterns")
    @skipIfA10G("failure on a10g")
    @skipIfTileIR("accuracy failure")
    def test_squeeze_and_excitation_net_bwd_da(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, b, c, d)
        expected = a_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_da",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.3,
        )

    @xfailIfCute("CuTe squeeze-and-excitation backward still fails lowering/runtime")
    @skipIfA10G("failure on a10g")
    @skipIfTileIR("accuracy failure")
    def test_squeeze_and_excitation_net_bwd_db(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        # Create configured kernel with explicit config
        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, d, c)
        expected = b_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_db",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.3,
        )

    def test_grpo_loss_fwd(self):
        """Test forward pass for GRPO loss."""
        B, L, V = 4, 512, 2048
        temperature = 0.9
        beta = 0.04
        eps_low = 0.2
        eps_high = 0.4

        torch.manual_seed(42)
        logits = torch.randn([B, L + 1, V], device=DEVICE, dtype=torch.bfloat16)
        completion_ids = torch.randint(0, V, (B, L), device=DEVICE, dtype=torch.int64)
        old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)
        completion_mask = torch.ones(B, L, device=DEVICE, dtype=torch.float32)

        from examples.grpo_loss import extract_selected_logits_pytorch

        selected_logits = extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )

        from examples.grpo_loss import torch_grpo_loss

        expected_loss, expected_kl, expected_clipped = torch_grpo_loss(
            logits.float(),
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        args = (
            logits,
            selected_logits,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        # grpo_loss_forward returns (loss, kl_loss, is_clipped, lse)
        # We only check loss, kl_loss, is_clipped (lse is None in expected)
        expected = (expected_loss, expected_kl, expected_clipped, None)

        check_example(
            "grpo_loss",
            args,
            expected,
            fn_name="grpo_loss_forward",
            rtol=1e-2,
            atol=1e-1,
            block_sizes=[4, 16, 16],
        )

    @xfailIfPallas("InductorLoweringError")
    def test_grpo_loss_bwd(self):
        """Test backward pass for GRPO loss."""
        B, L, V = 2, 64, 128
        temperature = 0.9
        beta = 0.04
        eps_low = 0.2
        eps_high = 0.4

        torch.manual_seed(42)
        logits = torch.randn(
            [B, L + 1, V], device=DEVICE, dtype=torch.bfloat16, requires_grad=True
        )
        completion_ids = torch.randint(0, V, (B, L), device=DEVICE, dtype=torch.int64)
        old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)
        completion_mask = torch.ones(B, L, device=DEVICE, dtype=torch.float32)

        # Pre-compute selected logits and run forward pass to get lse
        from examples.grpo_loss import extract_selected_logits_pytorch
        from examples.grpo_loss import grpo_loss_forward

        from helion._testing import code_and_output

        selected_logits = extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )

        forward_args = (
            logits,
            selected_logits,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        _, (_, _, _, lse) = code_and_output(
            grpo_loss_forward,
            forward_args,
            block_sizes=[4, 16, 16],
        )

        grad_output = torch.randn(B, L, device=DEVICE, dtype=torch.float32)

        logits_torch = logits.detach().clone().float().requires_grad_(True)
        from examples.grpo_loss import torch_grpo_loss

        loss_torch, _, _ = torch_grpo_loss(
            logits_torch,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )
        loss_torch.backward(grad_output)
        expected_grad = logits_torch.grad

        args = (
            grad_output,
            logits,
            selected_logits,
            completion_ids,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            lse,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        check_example(
            "grpo_loss",
            args,
            expected_grad,
            fn_name="grpo_loss_backward",
            rtol=1e-2,
            atol=1e-1,
            block_sizes=[4, 16, 16],
        )

    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    def test_broadcast_matmul(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=torch.float32),
            torch.randn([768, 1024], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "broadcast_matmul",
            args,
            torch.matmul(args[0], args[1]),
            block_sizes=[128, 128, 128],
        )

    @xfailIfCute("CuTe batched softmax example still fails CUTLASS DSL codegen")
    def test_batch_softmax(self):
        args = (torch.randn([16, 512, 1024], device=DEVICE, dtype=torch.bfloat16),)
        check_example(
            "batch_softmax",
            args,
            torch.nn.functional.softmax(args[0], dim=-1),
            block_sizes=[8],
        )

    @xfailIfCute(
        "CuTe batched softmax block-pointer example still fails CUTLASS DSL codegen"
    )
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_batch_softmax_block_ptr(self):
        args = (torch.randn([4, 128, 1024], device=DEVICE, dtype=torch.bfloat16),)
        check_example(
            "batch_softmax",
            args,
            torch.nn.functional.softmax(args[0], dim=-1),
            block_sizes=[8],
            indexing="block_ptr",
        )

    @xfailIfCute("CuTe GDN forward example still fails CUTLASS DSL codegen")
    @xfailIfPallas("operation not supported on TPU")
    def test_gdn_fwd_h(self):
        """Test gated delta net forward h kernel."""
        batch = 2
        nheads = 4
        seqlen = 512
        chunk_size = 64
        dhead = 16
        dstate = 32

        k = torch.randn(
            batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device=DEVICE
        )
        k = torch.nn.functional.rms_norm(k, (dhead,))
        w = torch.randn(
            batch,
            seqlen // chunk_size,
            chunk_size,
            nheads,
            dhead,
            dtype=torch.float32,
            device=DEVICE,
        )
        wu, ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
        w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
        w = (
            w.permute(0, 1, 3, 2, 4)
            .reshape(batch, seqlen, nheads, dhead)
            .to(torch.bfloat16)
        )
        u = torch.randn(
            batch, seqlen, nheads, dstate, dtype=torch.bfloat16, device=DEVICE
        )
        u = torch.nn.functional.rms_norm(u, (dstate,))
        g = torch.cumsum(
            0.5
            * math.log(1 / dhead)
            * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE),
            dim=1,
        )

        args = (k, w, u, g, chunk_size)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "gdn_fwd_h.py")
        expected = mod.ref_gdn_fwd_h(*args)

        check_example(
            "gdn_fwd_h",
            args,
            expected,
            fn_name="helion_gdn_fwd_h",
        )

    @skipIfRocm("default config exceeds thread limit on ROCm")
    @skipIfTileIR("PassManager::run failed on TileIR backend")
    def test_long_sum(self):
        args = (torch.randn([4, 130000], device=DEVICE, dtype=torch.float32),)
        check_example(
            "long_sum",
            args,
            args[0].sum(-1),
            fn_name="longsum",
        )

    @xfailIfPallas("JAX tracer error with dynamic shapes")
    def test_long_sum_looped(self):
        args = (torch.randn([4, 130000], device=DEVICE, dtype=torch.float32),)
        check_example(
            "long_sum",
            args,
            args[0].sum(-1),
            fn_name="longsum_w_red_loop",
            block_sizes=[1],
            reduction_loops=[32768],
        )

    @skipIfPallas("flex_attention requires torch.compile and closures")
    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe flex attention destabilizes later cute tests when it fails in-process",
    )
    @skipIfRefEager("scalar_prefetch indexing not supported in ref interpreter")
    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe flex attention example still returns incorrect results",
    )
    def test_flex_attention(self):
        z, h, n_ctx, head_dim = 2, 4, 256, 64
        q, k, v = [
            torch.randn((z, h, n_ctx, head_dim), dtype=HALF_DTYPE, device=DEVICE)
            for _ in range(3)
        ]

        mod = import_path(EXAMPLES_DIR / "flex_attention.py")
        # Set a fixed config to skip autotuning (exceeds CI timeout)
        config = helion.Config(block_sizes=[64, 64])
        mod.helion_flex_attention_kernel.configs = [config]
        out = mod.helion_flex_attention(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe Mamba2 chunk-state destabilizes later cute tests when it fails in-process",
    )
    @xfailIfPallas("BlockSpec tiling failure")
    def test_mamba2_chunk_state(self):
        batch, nheads, ngroups, seqlen, chunk_size, headdim, dstate = (
            2,
            8,
            1,
            256,
            64,
            32,
            16,
        )
        nchunks = seqlen // chunk_size
        B = torch.rand(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
        x = torch.rand(batch, seqlen, nheads, headdim, dtype=HALF_DTYPE, device=DEVICE)
        dt = torch.rand(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        dA_cumsum = torch.rand(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        args = (B, x, dt, dA_cumsum)

        mod = import_path(EXAMPLES_DIR / "mamba2_chunk_state.py")
        expected = mod.ref_chunk_state(*args)

        check_example(
            "mamba2_chunk_state",
            args,
            expected,
            fn_name="helion_mamba2_chunk_state_kernel",
            atol=0.1,
            rtol=0.1,
        )

    @xfailIfCute("CuTe Mamba2 chunk-scan example still returns incorrect results")
    @xfailIfPallas("BlockSpec tiling failure")
    def test_mamba2_chunk_scan(self):
        batch, nheads, ngroups, seqlen, chunk_size, headdim, dstate = (
            2,
            8,
            1,
            256,
            64,
            32,
            16,
        )
        nchunks = seqlen // chunk_size
        cb = torch.zeros(
            batch,
            nchunks,
            ngroups,
            chunk_size,
            chunk_size,
            dtype=HALF_DTYPE,
            device=DEVICE,
        )
        torch.manual_seed(0)
        x = torch.zeros(batch, seqlen, nheads, headdim, dtype=HALF_DTYPE, device=DEVICE)
        dt = torch.zeros(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        dA_cumsum = torch.zeros(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        # Keep the recurrence path deterministic and non-zero so backend bugs
        # cannot hide behind allocator state when output buffers are recycled.
        C = torch.randn(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
        prev_states = torch.randn(
            batch,
            nchunks,
            nheads,
            headdim,
            dstate,
            dtype=HALF_DTYPE,
            device=DEVICE,
        )
        D_param = torch.zeros(nheads, dtype=HALF_DTYPE, device=DEVICE)
        args = (cb, x, dt, dA_cumsum, C, prev_states, D_param)

        mod = import_path(EXAMPLES_DIR / "mamba2_chunk_scan.py")
        expected = mod.ref_chunk_scan(*args)

        check_example(
            "mamba2_chunk_scan",
            args,
            expected,
            fn_name="helion_mamba2_chunk_scan_kernel",
            atol=0.1,
            rtol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
