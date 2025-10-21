from __future__ import annotations

import ast
from collections import namedtuple
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from packaging import version
import pytest
import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion import _compat
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import PROJECT_ROOT
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRefEager
import helion.language as hl


class TestMisc(RefEagerTestBase, TestCase):
    def test_binary_operation_duplicate_args(self):
        """Test case to reproduce issue #221: binary operations with duplicate tensor references"""

        @helion.kernel(autotune_effort="none")
        def kernel_with_duplicate_refs(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                val = x[tile]
                result[tile] = (
                    val * val + val
                )  # Multiple uses of same variable - triggers the bug
            return result

        x = torch.randn([16, 16], device=DEVICE)
        expected = x * x + x

        code, result = code_and_output(kernel_with_duplicate_refs, (x,))
        torch.testing.assert_close(result, expected)

    def test_torch_alloc(self):
        @helion.kernel(config={"block_sizes": [64, 64]})
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = x.new_empty([m])
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m):
                acc = x.new_zeros([tile_m, block_size_n])
                for tile_n in hl.tile(n, block_size=block_size_n):
                    acc += x[tile_m, tile_n]
                out[tile_m] = acc.sum(dim=-1)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x.sum(-1), atol=1e-2, rtol=1e-2)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Decorator ordering checks not applicable in ref eager mode")
    def test_decorator(self):
        def mydec(func):
            return func

        @mydec
        @helion.kernel
        def add1(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel
        @mydec
        def add2(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @mydec
        @helion.kernel(config=helion.Config(block_size=[4]))
        def add3(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(4, device=DEVICE)

        code_and_output(add1, (x, x))

        with pytest.raises(
            expected_exception=helion.exc.DecoratorAfterHelionKernelDecorator,
            match="Decorators after helion kernel decorator are not allowed",
        ):
            code_and_output(add2, (x, x))

        code_and_output(add3, (x, x))

    @skipIfRefEager("Inductor lowering tests not applicable in ref eager mode")
    def test_patch_inductor_lowerings(self):
        if version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"):
            from helion._compiler.inductor_lowering_extra import (
                register_inductor_lowering,
            )
        else:
            from torch._inductor.lowering import (
                register_lowering as register_inductor_lowering,
            )

        from helion._compiler.inductor_lowering_extra import inductor_lowering_dispatch
        from helion._compiler.inductor_lowering_extra import patch_inductor_lowerings

        inductor_lowerings_orig = torch._inductor.lowering.lowerings.copy()

        @torch.library.custom_op("helion_test::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        # Case 1: Register new lowering for the custom op
        @register_inductor_lowering(
            torch.ops.helion_test.foo, lowering_dict=inductor_lowering_dispatch
        )
        def foo_lowering(x):
            return x

        # Case 2: Register a patched lowering for add.Tensor
        @register_inductor_lowering(
            torch.ops.aten.add.Tensor, lowering_dict=inductor_lowering_dispatch
        )
        def add_lowering(*args, **kwargs):
            pass

        # Check that within `patch_inductor_lowerings()` context manager, the patched lowerings are used.
        with patch_inductor_lowerings():
            assert torch.ops.helion_test.foo in torch._inductor.lowering.lowerings
            assert torch.ops.aten.add.Tensor in torch._inductor.lowering.lowerings
            assert (
                torch._inductor.lowering.lowerings[torch.ops.aten.add.Tensor]
                != inductor_lowerings_orig[torch.ops.aten.add.Tensor]
            )

        # Check that outside the context manager, the original lowerings are restored.
        assert len(torch._inductor.lowering.lowerings.keys()) == len(
            inductor_lowerings_orig.keys()
        )
        for op in torch._inductor.lowering.lowerings:
            assert torch._inductor.lowering.lowerings[op] == inductor_lowerings_orig[op]

    def test_inputs(self):
        @helion.kernel
        def kernel(a_list, b_dict, b_tuple, c_named_tuple, d_dataclass):
            a0, a1 = a_list
            b0 = b_dict["b0"]
            (b1,) = b_tuple
            c0, c1 = c_named_tuple.x, c_named_tuple.y
            d0, d1 = d_dataclass.x, d_dataclass.y

            o0, o1 = torch.empty_like(a0), torch.empty_like(a1)
            for tile in hl.tile(a0.size()):
                o0[tile] = a0[tile] + b0[tile] + c0[tile] + d0[tile]
                o1[tile] = a1[tile] + b1[tile] + c1[tile] + d1[tile]
            return [o0, o1]

        x = torch.ones(4, device=DEVICE)
        Point = namedtuple("Point", ["x", "y"])  # noqa: PYI024
        p = Point(x, x)

        @dataclass(frozen=True)
        class Point2:
            x: torch.Tensor
            y: torch.Tensor

        p2 = Point2(x, x)

        code, result = code_and_output(kernel, ([x, x], {"b0": x}, (x,), p, p2))
        torch.testing.assert_close(result[0], 4 * x)
        torch.testing.assert_close(result[1], 4 * x)
        self.assertExpectedJournal(code)

    def test_dtype_cast_preserved_before_second_dot(self):
        """Regression for issue #512: ensure p.to(v.dtype) is honored before a second dot.

        Pattern: qk = hl.dot(q, k, tf32) -> pointwise silu -> cast to v.dtype -> hl.dot(p, v)
        Previously, the cast could be hoisted/ignored leading to FP32 p fed into BF16 v.
        This test ensures kernel runs and matches reference with BF16 inputs.
        """

        @helion.kernel(autotune_effort="none", dot_precision="tf32")
        def kernel(
            q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
        ) -> torch.Tensor:
            # 2D dot test: q[M, K], k[K, N], v[N, H] -> out[M, H]
            m_dim, k_dim = q_in.size()
            k2_dim, n_dim = k_in.size()
            assert k2_dim == k_dim
            v2_dim, h_dim = v_in.size()
            h_dim = hl.specialize(h_dim)
            assert v2_dim == n_dim
            out = torch.empty([m_dim, h_dim], dtype=q_in.dtype, device=q_in.device)
            for tile_m in hl.tile(m_dim):
                acc = hl.zeros([tile_m, h_dim], dtype=torch.float32)
                q = q_in[tile_m, :]
                for tile_n in hl.tile(n_dim):
                    k = k_in[:, tile_n]
                    # First dot: accumulate in TF32 (fp32 compute)
                    qk = hl.dot(q, k)
                    # Apply SiLU = x * sigmoid(x) in pointwise ops
                    p = torch.sigmoid(qk)
                    p = qk * p
                    v = v_in[tile_n, :]
                    # Cast to match v's dtype (bf16)
                    p = p.to(v.dtype)
                    # Second dot
                    acc = hl.dot(p, v, acc=acc)
                out[tile_m, :] = acc.to(out.dtype)
            return out

        # Small sizes for quick runtime
        M, K, N, H = 32, 64, 32, 64
        q = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(N, H, device=DEVICE, dtype=torch.bfloat16)

        code, out = code_and_output(kernel, (q, k, v))

        # Reference computation in float32, with explicit bf16 cast for p
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        qk = qf @ kf  # [M, N]
        p = qk * torch.sigmoid(qk)
        p = p.to(torch.bfloat16).to(torch.float32)
        expected = p @ vf  # [M, H]
        expected = expected.to(out.dtype)

        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_config_flatten_issue(self):
        @helion.kernel(autotune_effort="none")
        def test_tile_begin(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.begin, tile_n.begin] = 1
            return out

        x = torch.randn(64, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16])
        test_tile_begin.bind((x,)).to_triton_code(config)
        result = test_tile_begin.bind((x,)).compile_config(config)(x)
        self.assertEqual(result.sum().item(), 16)

        @helion.kernel(autotune_effort="none")
        def test_tile_end(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.end, tile_n.end] = 1
            return out

        x = torch.randn(64, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16])
        test_tile_end.bind((x,)).to_triton_code(config)
        result = test_tile_end.bind((x,)).compile_config(config)(x)
        self.assertEqual(result.sum().item(), 12)

        @helion.kernel(autotune_effort="none")
        def test_tile_id(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.id, tile_n.id] = 1
            return out

        x = torch.randn(64, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16])
        test_tile_id.bind((x,)).to_triton_code(config)
        result = test_tile_id.bind((x,)).compile_config(config)(x)
        self.assertEqual(result.sum().item(), 16)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_block_size_constexpr_fix(self):
        """Test that tile.block_size can be used in expressions without compilation errors."""

        @helion.kernel(autotune_effort="none")
        def test_tile_block_size_usage(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.shape[0]):
                # This should not cause a compilation error when tile.block_size is used
                # in expressions that generate .to() calls
                block_size_temp = tile.block_size
                mask = tile.index % block_size_temp == block_size_temp - 1
                out[tile] = torch.where(mask, 1, 0)
            return out

        x = torch.randn(32, device=DEVICE)
        code, result = code_and_output(test_tile_block_size_usage, (x,))
        self.assertExpectedJournal(code)
        # The result should have 1s at positions that are last in their tile
        self.assertTrue(result.sum().item() > 0)

    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_to_triton_code_optional_config(self):
        """Test that to_triton_code() works without explicit config argument."""

        # Test 1: Kernel with single config - should use that config
        @helion.kernel(config={"block_sizes": [64]})
        def kernel_single_config(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] * 2
            return result

        x = torch.randn([32], device=DEVICE)
        bound_kernel = kernel_single_config.bind((x,))

        # Should work without config argument
        code_without_config = bound_kernel.to_triton_code()
        code_with_config = bound_kernel.to_triton_code({"block_sizes": [64]})
        self.assertEqual(code_without_config, code_with_config)

        # Test 2: Kernel with autotune_effort="none" - should use default config
        @helion.kernel(autotune_effort="none")
        def kernel_default_config(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] * 3
            return result

        bound_kernel_default = kernel_default_config.bind((x,))

        # Should work without config argument using default config
        code_default = bound_kernel_default.to_triton_code()
        self.assertIsInstance(code_default, str)
        self.assertIn("def", code_default)  # Basic sanity check

        # Test 3: Kernel with no configs and no default - should raise error
        @helion.kernel()
        def kernel_no_config(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] * 4
            return result

        bound_kernel_no_config = kernel_no_config.bind((x,))

        # Should raise RuntimeError when no implicit config available
        with self.assertRaises(RuntimeError) as cm:
            bound_kernel_no_config.to_triton_code()
        self.assertIn(
            "no config provided and no implicit config available", str(cm.exception)
        )

    def test_scalar_tensor_item_method(self):
        """Test using scalar_tensor.item() to extract scalar value in kernel"""

        @helion.kernel(autotune_effort="none")
        def kernel_with_scalar_item(
            x: torch.Tensor, scalar_tensor: torch.Tensor
        ) -> torch.Tensor:
            result = torch.empty_like(x)
            scalar_val = scalar_tensor.item()
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + scalar_val
            return result

        x = torch.randn(100, device=DEVICE)
        code, result = code_and_output(
            kernel_with_scalar_item, (x, torch.tensor(5.0, device=DEVICE))
        )
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, x + 5)

        code2, result2 = code_and_output(
            kernel_with_scalar_item, (x, torch.tensor(10.0, device=DEVICE))
        )
        self.assertEqual(code, code2)
        torch.testing.assert_close(result2, x + 10)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_tuple_literal_subscript(self):
        @helion.kernel
        def tuple_literal_index_kernel(inp_tuple) -> torch.Tensor:
            out = torch.empty_like(inp_tuple[0])
            for tile in hl.tile(out.size()):
                out[tile] = (inp_tuple[0][tile] + inp_tuple[1][tile]) * inp_tuple[2]
            return out

        inp_tuple = (
            torch.randn(8, 30, device=DEVICE, dtype=torch.float32),
            torch.randn(8, 32, device=DEVICE, dtype=torch.bfloat16),
            3,
        )
        code_pointer, result = code_and_output(
            tuple_literal_index_kernel,
            (inp_tuple,),
            block_size=[8, 8],
            indexing="pointer",
        )
        torch.testing.assert_close(result, (inp_tuple[0] + inp_tuple[1][:, :30]) * 3)

        code_block, result = code_and_output(
            tuple_literal_index_kernel,
            (inp_tuple,),
            block_size=[8, 8],
            indexing="block_ptr",
        )
        torch.testing.assert_close(result, (inp_tuple[0] + inp_tuple[1][:, :30]) * 3)

        self.assertNotEqualCode(code_pointer, code_block)
        self.assertExpectedJournal(code_pointer + code_block)

    @unittest.skipUnless(
        supports_tensor_descriptor(), "Tensor descriptor support is required"
    )
    def test_tuple_literal_subscript_w_descriptor(self):
        @helion.kernel
        def tuple_literal_index_kernel(inp_tuple) -> torch.Tensor:
            out = torch.empty_like(inp_tuple[0])
            for tile in hl.tile(out.size()):
                out[tile] = (inp_tuple[0][tile] + inp_tuple[1][tile]) * inp_tuple[2]
            return out

        inp_tuple = (
            torch.randn(8, 30, device=DEVICE, dtype=torch.float32),
            torch.randn(8, 32, device=DEVICE, dtype=torch.bfloat16),
            3,
        )
        code, result = code_and_output(
            tuple_literal_index_kernel,
            (inp_tuple,),
            block_size=[8, 8],
            indexing="tensor_descriptor",
        )
        torch.testing.assert_close(result, (inp_tuple[0] + inp_tuple[1][:, :30]) * 3)
        self.assertExpectedJournal(code)

    def test_tuple_unpack(self):
        @helion.kernel
        def tuple_unpack_kernel(inp_tuple) -> torch.Tensor:
            a, b, x = inp_tuple
            out = torch.empty_like(a)
            for tile in hl.tile(out.size(0)):
                out[tile] = a[tile] + b[tile] + x
            return out

        inp_tuple = (
            torch.randn(16, device=DEVICE, dtype=torch.float32),
            torch.randn(16, device=DEVICE, dtype=torch.bfloat16),
            5,
        )
        code, result = code_and_output(tuple_unpack_kernel, (inp_tuple,), block_size=4)
        torch.testing.assert_close(result, inp_tuple[0] + inp_tuple[1] + 5)

        self.assertExpectedJournal(code)

    def test_propagate_tile(self):
        @helion.kernel
        def copy_kernel(a: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)

            for tile in hl.tile(a.size(0), block_size=4):
                t1 = tile
                t2 = tile
                out[t2] = a[t1]
            return out

        args = (torch.randn(16, device=DEVICE, dtype=torch.bfloat16),)
        code, result = code_and_output(copy_kernel, args)
        torch.testing.assert_close(result, args[0])
        self.assertExpectedJournal(code)

    @parametrize("static_shapes", (True, False))
    def test_sequence_assert(self, static_shapes):
        @helion.kernel(static_shapes=static_shapes)
        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            assert a.size() == b.size()
            out = torch.empty_like(a)

            for tile in hl.tile(a.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(16, 1, device=DEVICE)
        code, result = code_and_output(kernel, (a, a))
        torch.testing.assert_close(result, a + a)
        self.assertExpectedJournal(code)

    @skipIfRefEager("no code execution")
    def test_triton_repro_add(self):
        mod = import_path(EXAMPLES_DIR / "add.py")
        a = torch.randn(16, 1, device=DEVICE)
        bound_kernel = mod.add.bind((a, a))
        code = bound_kernel.to_triton_code(
            config=bound_kernel.config_spec.default_config(), emit_repro_caller=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "test.py"
            tmp.write_text(code)
            result = subprocess.run(
                [sys.executable, str(tmp)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{PROJECT_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                },
            )
            self.assertEqual(result.returncode, 0, msg=f"stderr:\n{result.stderr}")
        self.assertExpectedJournal(code)

    @skipIfRefEager("no code execution")
    @parametrize("static_shapes", (True, False))
    def test_triton_repro_custom(self, static_shapes):
        @helion.kernel(static_shapes=static_shapes)
        def kernel(
            t: torch.Tensor, i: int, s: str, b: bool, f: float, zero_dim_t: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty_like(t)
            for tile in hl.tile(t.size()):
                if b and len(s) > 2:
                    out[tile] = t[tile] + i + f
            return out, zero_dim_t

        a = torch.randn(16, 1, device=DEVICE)
        bound_kernel = kernel.bind((a, 1, "foo", True, 1.2, a.sum()))
        code = bound_kernel.to_triton_code(
            config=bound_kernel.config_spec.default_config(), emit_repro_caller=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "test.py"
            tmp.write_text(code)
            result = subprocess.run(
                [sys.executable, str(tmp)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{PROJECT_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                },
            )
            self.assertEqual(
                result.returncode, 0, msg=f"code:{code}\nstderr:\n{result.stderr}"
            )
        self.assertExpectedJournal(code)

    @skipIfRefEager("no code execution")
    def test_repro_parseable(self):
        @helion.kernel
        def kernel(fn, t: torch.Tensor):
            a = torch.empty_like(t)
            for h in hl.tile(a.size(0)):
                a[h] = fn(t[h])
            return a

        bound_kernel = kernel.bind((lambda a: a, torch.ones(4, device=DEVICE)))
        code = bound_kernel.to_triton_code(
            config=bound_kernel.config_spec.default_config(), emit_repro_caller=True
        )
        ast.parse(code)


instantiate_parametrized_tests(TestMisc)


if __name__ == "__main__":
    unittest.main()
