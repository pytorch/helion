from __future__ import annotations

import unittest
from unittest.mock import patch

from packaging import version
import torch
import torch.nn.functional as F

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import check_example
from helion._testing import import_path
from helion._testing import skipIfA10G
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"


class TestExamples(RefEagerTestBase, TestCase):
    def test_add(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "add", args, sum(args), block_sizes=[128, 1], flatten_loop=True
            )
        )

    def test_matmul(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                args[0] @ args[1],
                block_sizes=[16, 16, 16],
                l2_grouping=4,
            )
        )

    def test_matmul_bwd(self):
        """Test backward pass for matmul computation."""
        # Create tensors with requires_grad=True like rms_norm_bwd test
        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        grad_out = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)

        # Compute expected gradients with PyTorch
        mat1_torch = mat1.detach().clone().requires_grad_(True)
        mat2_torch = mat2.detach().clone().requires_grad_(True)
        result_torch = torch.matmul(mat1_torch, mat2_torch)
        result_torch.backward(grad_out)

        args = (grad_out, mat1, mat2)

        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                (mat1_torch.grad, mat2_torch.grad),  # Expected: (grad_mat1, grad_mat2)
                fn_name="matmul_bwd",
                block_sizes=[
                    16,
                    16,
                    16,
                    16,
                    16,
                    16,
                ],  # [tile_m1, tile_k1, tile_n1, tile_k2, tile_n2, tile_m2]
            )
        )

    def test_addmm_bwd(self):
        """Test backward pass for addmm computation."""
        # Create tensors with requires_grad=True following the matmul_bwd pattern
        bias = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        grad_out = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        alpha = 1.0
        beta = 1.0

        # Compute expected gradients with PyTorch
        bias_torch = bias.detach().clone().requires_grad_(True)
        mat1_torch = mat1.detach().clone().requires_grad_(True)
        mat2_torch = mat2.detach().clone().requires_grad_(True)
        result_torch = torch.addmm(
            bias_torch, mat1_torch, mat2_torch, alpha=alpha, beta=beta
        )
        result_torch.backward(grad_out)

        args = (grad_out, bias, mat1, mat2, alpha, beta)

        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                (
                    bias_torch.grad,
                    mat1_torch.grad,
                    mat2_torch.grad,
                ),  # Expected: (grad_input, grad_mat1, grad_mat2)
                fn_name="addmm_bwd",
            )
        )

    @skipIfRocm("failure on rocm")
    def test_matmul_layernorm_static_shapes(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
        )
        self.assertExpectedJournal(
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
                static_shapes=True,
            )
        )

    @skipIfRocm("failure on rocm")
    def test_matmul_layernorm_dynamic_shapes(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
        )
        self.assertExpectedJournal(
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
        )

    @unittest.skipIf(
        version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"),
        "Requires torch 2.8+",
    )
    def test_bmm(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=torch.float16),
            torch.randn([16, 768, 1024], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "bmm",
                args,
                torch.bmm(args[0], args[1]),
                block_sizes=[16, 16, 16, 16],
            )
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "FP8 requires GPU with compute capability >= 9.0 (e.g., H100)",
    )
    @skipIfRocm("failure on rocm")
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

        self.assertExpectedJournal(
            check_example(
                "fp8_gemm",
                args,
                expected,
                block_sizes=[16, 16, 32],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_template_via_closure0(self):
        bias = torch.randn([1, 1024], device=DEVICE, dtype=torch.float16)
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul",
                block_sizes=[64, 64, 16],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="pointer",
                l2_grouping=64,
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("Failed on XPU - https://github.com/pytorch/helion/issues/795")
    def test_template_via_closure1(self):
        bias = torch.randn([1, 1024], device=DEVICE, dtype=torch.float16)
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul",
                block_sizes=[64, 64, 16],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="block_ptr",
                l2_grouping=64,
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_template_via_closure2(self):
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda x, _: torch.nn.functional.relu(x),
        )
        self.assertExpectedJournal(
            check_example(
                "matmul",
                args,
                torch.relu(args[0] @ args[1]),
                fn_name="matmul",
                block_sizes=[64, 64, 16],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="block_ptr",
                l2_grouping=64,
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_softmax(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_softmax_looped(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
                reduction_loop=32,
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_softmax_decomposed(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_decomposed",
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
            )
        )

    def test_softmax_two_pass(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_two_pass",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_softmax_two_pass_block_ptr(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_two_pass",
                block_sizes=[8, 64],
                indexing="block_ptr",
            )
        )

    def test_cross_entropy(self):
        n, v = 128, 1000
        args = (
            torch.randn(n, v, device=DEVICE, dtype=torch.float32),
            torch.randint(0, v, (n,), device=DEVICE, dtype=torch.long),
        )
        self.assertExpectedJournal(
            check_example(
                "cross_entropy",
                args,
                torch.nn.functional.cross_entropy(*args),
            )
        )

    def test_welford(self):
        s, d = 128, 1024
        weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

        self.assertExpectedJournal(
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
        )

    def test_low_mem_dropout(self):
        from examples.low_mem_dropout import low_mem_dropout
        from examples.low_mem_dropout import low_mem_dropout_bwd

        from helion._testing import code_and_output

        p = 0.25
        size = 8192
        seed = 123
        seed2 = 456
        x = torch.randn(size=(size,)).to(device=DEVICE)

        _, out_fwd = code_and_output(
            low_mem_dropout,
            (p, x, seed),
        )

        grad_y = torch.ones_like(x)
        _, grad_x = code_and_output(
            low_mem_dropout_bwd,
            (p, grad_y, seed),
        )

        _, grad_x2 = code_and_output(
            low_mem_dropout_bwd,
            (p, grad_y, seed2),
        )

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

        self.assertExpectedJournal(
            check_example("low_mem_dropout", (p, grad_y, seed), grad_x),
        )

    @skipIfRocm("precision differences with bf16xint16 operations on rocm")
    @skipIfXPU("precision differences with bf16xint16 operations on xpu")
    def test_bf16xint16(self):
        from examples.bf16xint16_gemm import reference_bf16xint16_pytorch

        m, k, n = 65536, 1024, 1280

        x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)

        self.assertExpectedJournal(
            check_example(
                "bf16xint16_gemm",
                (x, w),
                reference_bf16xint16_pytorch(x, w, False),
                fn_name="_bf16xint16_gemm",
            )
        )

        x_int16 = torch.randint(
            -(2**15), 2**15 - 1, (m, k), device=DEVICE, dtype=torch.int16
        )
        w_bf16 = torch.randn([k, n], device=DEVICE, dtype=torch.bfloat16)

        self.assertExpectedJournal(
            check_example(
                "bf16xint16_gemm",
                (x_int16, w_bf16),
                reference_bf16xint16_pytorch(x_int16, w_bf16, True),
                fn_name="_int16xbf16_gemm",
            )
        )

    def test_rms_norm_fwd(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float16),
            torch.randn([256], device=DEVICE, dtype=torch.float16),
            1e-5,
        )
        # Import and use the reference implementation from rms_norm.py
        mod = import_path(EXAMPLES_DIR / "rms_norm.py")
        expected = mod.rms_norm_pytorch(*args)

        self.assertExpectedJournal(
            check_example(
                "rms_norm",
                args,
                (expected, None),  # Expected: (output, 1/rms)
                fn_name="rms_norm_fwd",
                block_sizes=[16],
                indexing="pointer",
            )
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

        self.assertExpectedJournal(
            check_example(
                "swiglu",
                args,
                (x1.grad, x2.grad),
                fn_name="swiglu_bwd",
            )
        )

    def test_rms_norm_bwd(self):
        """Test backward pass for rms norm weight gradient."""
        batch_size, dim = 32, 64
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float16)
        weight = torch.randn(
            [dim], device=DEVICE, dtype=torch.float16, requires_grad=True
        )
        grad_out = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.float16)
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
        self.assertExpectedJournal(
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
        )

    def test_embedding_pointers(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "embedding",
                args,
                torch.nn.functional.embedding(*args),
                block_sizes=[1, 256],
                indexing="pointer",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_embedding_block_ptr(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "embedding",
                args,
                torch.nn.functional.embedding(*args),
                block_sizes=[8, 64],
                indexing="block_ptr",
                pid_type="xyz",
            )
        )

    @skipIfRocm("failure on rocm")
    def test_attention_pointer(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[1, 64, 32],
                indexing="pointer",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    def test_attention_block_pointer(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[16, 32, 16],
                num_stages=1,
                indexing="block_ptr",
            )
        )

    def test_attention_dynamic(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                fn_name="attention_dynamic",
                block_sizes=[1, 64, 32],
            )
        )

    def test_concat(self):
        args = (
            torch.randn(512, 500, device=DEVICE),
            torch.randn(512, 512, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "concatenate",
                args,
                torch.cat(args, dim=1),
                fn_name="concat2d_dim1",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_concat_block_ptr(self):
        args = (
            torch.randn(222, 100, device=DEVICE),
            torch.randn(222, 151, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "concatenate",
                args,
                torch.cat(args, dim=1),
                fn_name="concat2d_dim1",
                indexing="block_ptr",
                block_sizes=[128, 64],
            )
        )

    def test_jagged_dense_add(self):
        mod = import_path(EXAMPLES_DIR / "jagged_dense_add.py")
        args = (
            *mod.random_jagged_2d(500, 5000, device=DEVICE),
            torch.randn(500, 5000, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "jagged_dense_add",
                args,
                mod.jagged_dense_add_2d_reference(*args),
                fn_name="jagged_dense_add_2d",
            )
        )

    @skipIfRefEager("Test has skip_accuracy=True and doesn't call assert_close")
    def test_moe_matmul_ogs(self):
        mod = import_path(EXAMPLES_DIR / "moe_matmul_ogs.py")

        B = 1000  # tokens / rows
        K = 500  # hidden size
        N = 200  # output size
        n_experts = 30
        A = torch.randn(B, K, device=DEVICE, dtype=torch.float16)
        W = torch.randn(n_experts, K, N, device=DEVICE, dtype=torch.float16)
        top1_expert_per_token = torch.randint(n_experts, (B,), device=DEVICE)

        args = (A, W, top1_expert_per_token)
        helion_kernel_args = mod.moe_matmul_ogs_helion_kernel_args_gen(
            A, W, top1_expert_per_token
        )
        self.assertExpectedJournal(
            check_example(
                "moe_matmul_ogs",
                helion_kernel_args,
                mod.moe_matmul_ogs_reference(*args),
                block_sizes=[16, 16, 16],
                skip_accuracy=True,  # TODO(yf225): fix unstable numerics
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_matmul_split_k(self):
        args = (
            torch.randn(64, 1024, device=DEVICE),
            torch.randn(1024, 64, device=DEVICE),
        )
        self.assertExpectedJournal(
            check_example(
                "matmul_split_k",
                args,
                torch.matmul(*args),
                indexing="block_ptr",
                block_sizes=[16, 16, 32],
                split_k=8,
            )
        )

    def test_sum(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedJournal(
            check_example(
                "sum",
                args,
                torch.sum(args[0], dim=-1),
                fn_name="sum_kernel",
                block_sizes=[1],
                reduction_loops=[32768],
            )
        )

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

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_mean.py")
        expected = mod.reference_jagged_mean_kernel_pytorch(
            x_data, x_offsets, feature_counts, M
        )

        self.assertExpectedJournal(
            check_example(
                "jagged_mean",
                args,
                expected,
                fn_name="jagged_mean_kernel",
                block_sizes=[16, 8, 16],
            )
        )

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

        self.assertExpectedJournal(
            check_example(
                "segment_reduction",
                args,
                expected,
                fn_name="segmented_reduction_helion",
            )
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    def test_attention_persistent_interleaved_l2_grouping(self):
        """Test attention with persistent interleaved execution and L2 grouping for optimal performance."""
        args = (
            torch.randn(2, 16, 512, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=torch.float16, device=DEVICE),
        )

        self.assertExpectedJournal(
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
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "FP8 requires GPU with compute capability >= 9.0 (e.g., H100)",
    )
    @skipIfRocm("failure on rocm")
    def test_fp8_attention(self):
        batch = 2
        heads = 4
        seq_len = 256
        head_dim = 64

        # Create FP16 tensors
        q = torch.randn(
            batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE
        )

        # Import the module
        mod = import_path(EXAMPLES_DIR / "fp8_attention.py")

        # Prepare FP8 inputs using the module's preprocessing function
        q_fp8, k_fp8, v_fp8 = mod.preprocess_fp8_attention_inputs(q, k, v)
        args = (q_fp8, k_fp8, v_fp8, batch, heads)

        # Get expected output from kernel
        expected = mod.fp8_attention_pytorch(q, k, v)()

        self.assertExpectedJournal(
            check_example(
                "fp8_attention",
                args,
                expected,
                fn_name="fp8_attention_kernel",
                block_sizes=[64, 64],
                atol=0.2,
                rtol=0.1,
            )
        )

    def test_layernorm_with_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.float16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.float16)
        bias = torch.randn([64], device=DEVICE, dtype=torch.float16)

        args = (x, [64], weight, bias)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        self.assertExpectedJournal(
            check_example(
                "layer_norm",
                args,
                (expected_out, None, None),  # Expected: (output, mean, rstd)
                fn_name="layer_norm_fwd",
                block_size=32,
                num_warps=4,
                num_stages=3,
            )
        )

    def test_layernorm_no_bias(self):
        """Test forward pass for layer normalization without bias."""
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.float16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.float16)

        args = (x, [64], weight, None)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        self.assertExpectedJournal(
            check_example(
                "layer_norm",
                args,
                (expected_out, None, None),  # Expected: (output, mean, rstd)
                fn_name="layer_norm_fwd",
                block_size=32,
                num_warps=4,
                num_stages=3,
            )
        )

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
                [batch_size, dim], device=DEVICE, dtype=torch.float16
            )
            weight = torch.randn([dim], device=DEVICE, dtype=torch.float16)
            bias = torch.randn([dim], device=DEVICE, dtype=torch.float16)
            grad_out = torch.randn(
                [batch_size, dim], device=DEVICE, dtype=torch.float16
            )

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

            journal = check_example(
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
            if idx == 0:
                self.assertExpectedJournal(journal)

    def test_softmax_bwd(self):
        m, n = 2048, 2048
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.float16)

        from examples.softmax import softmax_two_pass

        config = helion.Config(block_size=[128, 128], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(softmax_two_pass.fn, config=config)
        y = configured_kernel(x)

        x_torch = x.detach().clone().requires_grad_(True)
        y_torch = torch.nn.functional.softmax(x_torch, dim=-1)
        y_torch.backward(grad_out)

        self.assertExpectedJournal(
            check_example(
                "softmax",
                (grad_out, y),
                x_torch.grad,
                fn_name="softmax_bwd",
                rtol=1e-3,
                atol=1e-3,
            )
        )

    def test_layernorm_without_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.float16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.float16)

        args = (x, [64], weight, None)
        # Test returns (output, mean, rstd) tuple
        expected_out = torch.nn.functional.layer_norm(x, [64], weight)
        expected = (expected_out, None, None)
        self.assertExpectedJournal(
            check_example(
                "layer_norm",
                args,
                expected,
                fn_name="layer_norm_fwd",
                block_size=32,
                num_warps=4,
                num_stages=3,
            )
        )

    @skipIfRefEager("ref eager mode hits CUDA indexing error with hl.store")
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

        self.assertExpectedJournal(
            check_example(
                "jagged_softmax",
                args,
                expected,
                fn_name="jagged_softmax_kernel",
                block_sizes=[16, 8, 16, 16],
            )
        )

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

        self.assertExpectedJournal(
            check_example(
                "jagged_hstu_attn",
                args,
                expected,
                fn_name="_helion_jagged_attention_kernel",
                block_sizes=[16, 16],
                atol=1e-2,
                rtol=1e-2,
            )
        )

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
        self.assertExpectedJournal(
            check_example(
                "grouped_gemm",
                args,
                expected,
                fn_name="grouped_gemm_jagged",
            )
        )

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
        self.assertExpectedJournal(
            check_example(
                "grouped_gemm",
                args,
                expected,
                fn_name="grouped_gemm_jagged_persistent",
            )
        )

    def test_geglu(self):
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "geglu",
                args,
                torch.nn.functional.gelu(args[0], approximate="tanh") * args[1],
                block_sizes=[16],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_swiglu(self):
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedJournal(
            check_example(
                "swiglu",
                args,
                torch.nn.functional.silu(args[0]) * args[1],
                fn_name="swiglu_fwd",
                block_sizes=[16],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_jsd(self):
        args = (
            torch.randn(
                [4 * 2048, 4096], device=DEVICE, dtype=torch.float32
            ).log_softmax(dim=-1),
            torch.randn(
                [4 * 2048, 4096], device=DEVICE, dtype=torch.float32
            ).log_softmax(dim=-1),
            None,
        )

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jsd.py")
        expected = mod.TorchJSDBaseline()
        self.assertExpectedJournal(
            check_example(
                "jsd",
                args,
                (expected(*args), None),
                fn_name="jsd_forward",
                block_sizes=[1, 4096],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_kl_div(self):
        args = (
            torch.randn(
                [8 * 512, 4096], device=DEVICE, dtype=torch.float32
            ).log_softmax(dim=-1),
            torch.randn([8 * 512, 4096], device=DEVICE, dtype=torch.float32).softmax(
                dim=-1
            ),
        )
        torch_kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(
            device=DEVICE
        )
        self.assertExpectedJournal(
            check_example(
                "kl_div",
                args,
                torch_kl_div(*args),
                fn_name="kl_div_forward",
                block_sizes=[1, 4096],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_gather_gemv(self):
        args = (
            torch.randn([8, 1024, 1024], device=DEVICE, dtype=torch.float32),
            torch.randint(0, 8, [2], device=DEVICE, dtype=torch.int32),
            torch.randn([1024], device=DEVICE, dtype=torch.float32),
        )

        def expected(w, idx, x):
            return w[idx].to(x.dtype) @ x

        check_example(
            "gather_gemv",
            args,
            expected(*args),
            fn_name="gather_gemv",
            block_sizes=[16, 16],
            num_warps=8,
            num_stages=1,
        )

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

        self.assertExpectedJournal(
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
        )

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

        self.assertExpectedJournal(
            check_example(
                "jagged_sum",
                args,
                expected,
                fn_name="jagged_sum_kernel",
                block_sizes=[16, 8, 16],
            )
        )

    def test_fused_linear_jsd(self):
        beta = 0.5
        ignore_index = 1
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
        expected = mod.fused_linear_jsd_pytorch(
            *args[:-2], student_input, teacher_input, student_weight, teacher_weight
        )

        self.assertExpectedJournal(
            check_example(
                "fused_linear_jsd",
                args,
                expected,
                fn_name="fused_linear_jsd_kernel",
                block_sizes=[32],
            )
        )

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

        self.assertExpectedJournal(
            check_example(
                "jagged_layer_norm",
                args,
                expected,
                fn_name="jagged_layer_norm_kernel",
                block_sizes=[4, 8, 8, 8, 8, 8, 8],
            )
        )

    def test_exp_fwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.float16)
        args = (x,)
        self.assertExpectedJournal(
            check_example(
                "exp",
                args,
                torch.exp(x),
                fn_name="exp_fwd",
                block_sizes=[16],
                num_warps=4,
                num_stages=3,
            )
        )

    def test_exp_bwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.float16).requires_grad_(True)
        y = torch.exp(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        torch_out = x.grad
        args = (
            grad_out,
            y,
        )
        self.assertExpectedJournal(
            check_example(
                "exp",
                args,
                torch_out,
                fn_name="exp_bwd",
                block_sizes=[16],
                num_warps=4,
                num_stages=3,
            )
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    def test_squeeze_and_excitation_net_fwd(self):
        m, n, k = 1024, 1024, 1024
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float16)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        args = (x, a, b)

        expected_out = torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b))
        c = torch.relu(x @ a)
        d = torch.sigmoid(c @ b)

        self.assertExpectedJournal(
            check_example(
                "squeeze_and_excitation_net",
                args,
                (expected_out, c, d),
                fn_name="squeeze_and_excitation_net_fwd",
                block_sizes=[16, 16, 16, 16],
                num_warps=4,
                num_stages=2,
            )
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    def test_squeeze_and_excitation_net_bwd_dx(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float16)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.float16)

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

        self.assertExpectedJournal(
            check_example(
                "squeeze_and_excitation_net",
                args,
                expected,
                fn_name="squeeze_and_excitation_net_bwd_dx",
                block_sizes=[16, 16, 16],
                num_warps=4,
                num_stages=2,
            )
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    def test_squeeze_and_excitation_net_bwd_da(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float16)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.float16)

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

        self.assertExpectedJournal(
            check_example(
                "squeeze_and_excitation_net",
                args,
                expected,
                fn_name="squeeze_and_excitation_net_bwd_da",
                block_sizes=[16, 16, 16],
                num_warps=4,
                num_stages=2,
            )
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    def test_squeeze_and_excitation_net_bwd_db(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float16)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        # Create configured kernel with explicit config
        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.float16)

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

        self.assertExpectedJournal(
            check_example(
                "squeeze_and_excitation_net",
                args,
                expected,
                fn_name="squeeze_and_excitation_net_bwd_db",
                block_sizes=[16, 16, 16],
                num_warps=4,
                num_stages=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
