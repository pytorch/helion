from __future__ import annotations

import unittest

from packaging import version
import torch

from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import check_example
from helion._testing import import_path
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm

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

    def test_rms_norm(self):
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
                expected,
                block_sizes=[16],
                indexing="pointer",
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
                block_sizes=[64, 64],
                indexing="pointer",
            )
        )

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
                block_sizes=[128, 64],
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
        max_M_tensor = torch.empty(M, device=DEVICE)

        args = (x_data, x_offsets, feature_counts, max_M_tensor)

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
                block_sizes=[64, 64],
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

    def test_layernorm(self):
        x = torch.randn([32, 64], device=DEVICE, dtype=torch.float16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.float16)
        bias = torch.randn([64], device=DEVICE, dtype=torch.float16)

        args = (x, [64], weight, bias)

        self.assertExpectedJournal(
            check_example(
                "layer_norm",
                args,
                torch.nn.functional.layer_norm(*args),
                fn_name="layer_norm_fwd",
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


if __name__ == "__main__":
    unittest.main()
