from __future__ import annotations

import unittest

from packaging import version
import torch

from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import check_example
from helion._testing import import_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TestExamples(TestCase):
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

    def test_template_via_closure0(self):
        bias = torch.randn([1, 1024], device=DEVICE, dtype=torch.float16)
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        self.assertExpectedJournal(
            check_example(
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul_with_epilogue",
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
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul_with_epilogue",
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
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1]),
                fn_name="matmul_with_epilogue",
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

    def test_rms_norm(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float16),
            torch.randn([256], device=DEVICE, dtype=torch.float16),
            1e-5,
        )
        # Reference implementation from rms_norm.py
        x, weight, eps = args
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        expected = weight * hidden_states.to(input_dtype)

        self.assertExpectedJournal(
            check_example(
                "rms_norm",
                args,
                expected,
                block_sizes=[16, 1],
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


if __name__ == "__main__":
    unittest.main()
