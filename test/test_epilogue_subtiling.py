from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


def _supports_epilogue_subtile_autotune() -> bool:
    return supports_tensor_descriptor() and torch.cuda.get_device_capability() >= (
        10,
        0,
    )


@helion.kernel
def matmul_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel
def matmul_with_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc + bias[tile_n]
    return out


@helion.kernel
def matmul_with_cast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out


@onlyBackends(["triton"])
class TestEpilogueSubtiling(TestCase):
    @skipIfRefEager("test checks generated Triton code")
    def test_codegen_s2(self):
        """S=2 produces tl.split and 2 stores for both bias-add and cast epilogues."""
        bias_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, bias_args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        torch.testing.assert_close(
            output, bias_args[0] @ bias_args[1] + bias_args[2], atol=1e-1, rtol=1e-2
        )
        self.assertIn("tl.split", code)
        self.assertEqual(code.count("tl.store("), 2)
        self.assertNotIn("tl.join", code)

        cast_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_cast, cast_args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        torch.testing.assert_close(
            output,
            (cast_args[0] @ cast_args[1]).to(torch.float16),
            atol=1e-1,
            rtol=1e-2,
        )
        self.assertIn("tl.split", code)
        self.assertEqual(code.count("tl.store("), 2)

    @skipIfRefEager("test checks generated Triton code")
    def test_codegen_s4(self):
        """S=4 produces multiple tl.split calls and 4 stores."""
        bias_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, bias_args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        torch.testing.assert_close(
            output, bias_args[0] @ bias_args[1] + bias_args[2], atol=1e-1, rtol=1e-2
        )
        self.assertGreater(code.count("tl.split"), 1)
        self.assertEqual(code.count("tl.store("), 4)
        self.assertNotIn("tl.join", code)

        cast_args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_cast, cast_args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        torch.testing.assert_close(
            output,
            (cast_args[0] @ cast_args[1]).to(torch.float16),
            atol=1e-1,
            rtol=1e-2,
        )
        self.assertGreater(code.count("tl.split"), 1)
        self.assertEqual(code.count("tl.store("), 4)

    def test_disabled_by_default(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(matmul_simple, args, block_sizes=[64, 64, 64])
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertNotIn("tl.split", code)

    @skipIfRefEager("test checks generated Triton code")
    def test_bool_true_normalizes_to_s2(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=True
        )
        torch.testing.assert_close(
            output, args[0] @ args[1] + args[2], atol=1e-1, rtol=1e-2
        )
        self.assertIn("tl.split", code)

    def test_numerical_correctness_s2_vs_s4(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256], device=DEVICE, dtype=torch.float32),
        )
        _, out_none = code_and_output(matmul_with_bias, args, block_sizes=[64, 64, 64])
        _, out_s2 = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=2
        )
        _, out_s4 = code_and_output(
            matmul_with_bias, args, block_sizes=[64, 64, 64], epilogue_subtile=4
        )
        torch.testing.assert_close(out_none, out_s2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_none, out_s4, atol=1e-5, rtol=1e-5)

    def test_autotune_field_enabled_for_large_k(self):
        args = (
            torch.randn([128, 1024], device=DEVICE, dtype=torch.float32),
            torch.randn([1024, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        if _supports_epilogue_subtile_autotune():
            self.assertIn("epilogue_subtile", fields)
            fragment = fields["epilogue_subtile"]
            assert isinstance(fragment, EnumFragment)
            self.assertEqual(fragment.choices, (None, 2))
        else:
            self.assertNotIn("epilogue_subtile", fields)

    def test_autotune_field_disabled_for_small_k(self):
        args = (
            torch.randn([128, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        self.assertNotIn("epilogue_subtile", fields)

    def test_autotune_field_large_k_allows_s4_on_blackwell(self):
        args = (
            torch.randn([128, 16384], device=DEVICE, dtype=torch.float32),
            torch.randn([16384, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        fields = matmul_with_bias.bind(args).config_spec._flat_fields()
        if not _supports_epilogue_subtile_autotune():
            self.assertNotIn("epilogue_subtile", fields)
            return

        self.assertIn("epilogue_subtile", fields)
        fragment = fields["epilogue_subtile"]
        assert isinstance(fragment, EnumFragment)
        expected_choices = (
            (None, 2, 4) if torch.cuda.get_device_capability() >= (10, 0) else (None, 2)
        )
        self.assertEqual(fragment.choices, expected_choices)


if __name__ == "__main__":
    unittest.main()
