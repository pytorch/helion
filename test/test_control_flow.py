from __future__ import annotations

import ast
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.device_ir import IfGraphInfo
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfPallas
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import xfailIfPallas
import helion.language as hl


@onlyBackends(["triton", "pallas", "cute"])
class TestControlFlow(RefEagerTestBase, TestCase):
    def test_if_arg(self):
        @helion.kernel()
        def fn(x, v):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code0, result = code_and_output(
            fn,
            (x, 5),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        code1, result = code_and_output(
            fn,
            (x, 10),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertEqual(code0, code1)

    @skipIfTileIR("result tensor mismatch in tileIR")
    def test_if_branch_modifies_nonlocal_var(self):
        @helion.kernel()
        def fn(x, v):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                delta = torch.zeros_like(x[tile])
                if 3 < v < 7:
                    delta += 1.0
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    delta += 2.0
                    out[tile] = torch.sin(x[tile])
                out[tile] = out[tile] + delta
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code0, result = code_and_output(
            fn,
            (x, 5),
        )
        torch.testing.assert_close(result, torch.sigmoid(x) + 1.0)
        code1, result = code_and_output(
            fn,
            (x, 10),
        )
        torch.testing.assert_close(result, torch.sin(x) + 2.0)
        self.assertEqual(code0, code1)

    def test_if_arg_indexed_scalar(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)

            for idx in hl.grid(x.shape[0]):
                # Since `y[idx]` is a scalar, comparing it against 0 will also create a scalar.
                if y[idx] != 0:
                    output[idx] = x[idx] * 2
                else:
                    output[idx] = x[idx]

            return output

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([1.0, 4.0, 3.0, 8.0], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, y),
        )
        torch.testing.assert_close(result, expected)

    @skipIfPallas("requires per-element tiling unavailable for small 1D tensors on TPU")
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_if_arg_tensor_sum(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)

            for tile in hl.tile(x.shape[0]):
                # Since `y[idx]` is a tensor, comparing it against 0 will also create a tensor.
                # if condition must takes a scalar, therefore we call .sum() to reduce the tensor to a scalar.
                if (y[tile] != 0).sum():
                    output[tile] = x[tile] * 2
                if (
                    y[tile] == 0
                ).sum():  # TODO(yf225): `else:` raises MLIR error in Triton, so we use a second if.
                    output[tile] = x[tile]

            return output

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([1.0, 4.0, 3.0, 8.0], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, y),
            block_size=1,
        )
        torch.testing.assert_close(result, expected)

    @skipIfPallas("requires per-element tiling unavailable for small 1D tensors on TPU")
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_if_stores_to_host_tensor_list(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = [torch.zeros_like(x), torch.zeros_like(x)]
            for tile in hl.tile(x.shape[0]):
                if (y[tile] != 0).sum():
                    for output in outputs:
                        output[tile] = x[tile] * 2
            return outputs[0], outputs[1]

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([0.0, 4.0, 0.0, 8.0], device=DEVICE)
        code, (first, second) = code_and_output(fn, (x, y), block_size=1)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_constant_true(self):
        @helion.kernel(
            config={
                "block_sizes": [128, 1],
                "flatten_loop": True,
                "indexing": "block_ptr",
            }
        )
        def fn(x):
            out = torch.empty_like(x)
            v = 4
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_constant_false(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            out = torch.empty_like(x)
            v = 15
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sin(x))

    def test_constant_branch_true_simple(self):
        @helion.kernel()
        def fn(x):
            out = torch.empty_like(x)
            v = 4
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, torch.sigmoid(x), rtol=1e-5, atol=1e-5)

    def test_constant_branch_false_simple(self):
        @helion.kernel()
        def fn(x):
            out = torch.empty_like(x)
            v = 15
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, torch.sin(x), rtol=1e-5, atol=1e-5)

    def test_if_arg_side_effect(self):
        """Test that lax.cond is used for scalar predicates with side effects.

        Both branches write to different outputs — jnp.where would incorrectly
        execute both branches, but lax.cond only executes the taken branch.
        """

        @helion.kernel()
        def fn(x: torch.Tensor, v: int) -> tuple[torch.Tensor, torch.Tensor]:
            out_a = torch.zeros_like(x)
            out_b = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                if v > 0:
                    out_a[tile] = x[tile] + 1.0
                else:
                    out_b[tile] = x[tile] + 2.0
            return out_a, out_b

        x = torch.randn([512, 512], device=DEVICE)

        # v=1: only out_a should be written
        code, (out_a, out_b) = code_and_output(fn, (x, 1))
        torch.testing.assert_close(out_a, x + 1.0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out_b, torch.zeros_like(x), rtol=1e-5, atol=1e-5)

        # v=-1: only out_b should be written
        code, (out_a, out_b) = code_and_output(fn, (x, -1))
        torch.testing.assert_close(out_a, torch.zeros_like(x), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out_b, x + 2.0, rtol=1e-5, atol=1e-5)

    @skipIfPallas("atomic_add not supported on Pallas")
    def test_error_in_non_taken_branch(self):
        def mul_relu_block_back_spec(x, y, dz):
            z = torch.relu(x * y[:, None])
            grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
            return grad_x, grad_y

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def mul_relu_block_backward_kernel(
            x: torch.Tensor,
            y: torch.Tensor,
            dz: torch.Tensor,
            use_atomics: hl.constexpr = False,
        ):
            # Get tensor sizes
            m, n = x.shape
            # Create output tensor for gradients
            dx = torch.empty_like(x)

            if use_atomics:
                dy = torch.zeros_like(y)
            else:
                dy = torch.empty_like(x)

            # Use Helion to tile the computation
            for tile_i, tile_j in hl.tile([m, n]):
                # Get input tiles
                x_tile = x[tile_i, tile_j]
                y_tile = y[tile_i]
                dz_tile = dz[tile_i, tile_j]

                # For ReLU, gradient is 1 where input > 0, 0 otherwise
                relu_mask = (x_tile * y_tile[:, None]) > 0
                # Chain rule: dx = dz * relu_grad * y
                relu_grad = torch.where(relu_mask, 1, 0)
                dx[tile_i, tile_j] = dz_tile * relu_grad * y_tile[:, None]

                # Chain rule: dy = dz * relu_grad * x -> backwards of broadcast(sum)
                if use_atomics:
                    local_dy_grad = torch.sum(dz_tile * relu_grad * x_tile, dim=1)
                    hl.atomic_add(dy, [tile_i], local_dy_grad)
                else:
                    local_dy_grad = dz_tile * relu_grad * x_tile
                    dy[tile_i, tile_j] = local_dy_grad

            if use_atomics:
                return dx, dy
            return dx, dy.sum(axis=-1)

        x = torch.randn(512, 1024, device=DEVICE, requires_grad=True)
        y = torch.randn(512, device=DEVICE, requires_grad=True)
        dz = torch.randn(512, 1024, device=DEVICE)
        expected = mul_relu_block_back_spec(x, y, dz)
        torch.testing.assert_close(
            mul_relu_block_backward_kernel(x, y, dz, False),
            expected,
            atol=1e-4,
            rtol=1e-4,
        )
        code, output = code_and_output(
            mul_relu_block_backward_kernel,
            (x, y, dz, True),
        )
        torch.testing.assert_close(
            output,
            expected,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_if_new_variable_in_static_range(self):
        """Test that variables defined inside if/else within static_range work correctly.

        When a host bool gates an if/else inside hl.static_range, the variable
        assigned in the branches must be available after the if/else.
        Regression test for https://github.com/pytorch/helion/issues/1584
        """

        @helion.kernel()
        def fn(x: torch.Tensor, flag: bool) -> torch.Tensor:
            T, D = x.shape
            y = torch.empty_like(x)
            for tile_t, tile_d in hl.tile([T, D]):
                acc = hl.zeros([tile_t, tile_d], dtype=torch.float32)
                for iw in hl.static_range(2):
                    if flag and iw == 0:
                        val = x[tile_t, tile_d].to(torch.float32) + 1
                    else:
                        val = x[tile_t, tile_d].to(torch.float32)
                    acc = acc + val
                y[tile_t, tile_d] = acc.to(y.dtype)
            return y

        x = torch.randn(4, 64, dtype=torch.bfloat16, device=DEVICE)

        # flag=True: first iteration adds 1, second doesn't
        code, result = code_and_output(fn, (x, True))
        expected = (x.float() + 1 + x.float()).to(x.dtype)
        torch.testing.assert_close(result, expected)

        # flag=False: neither iteration adds 1
        code, result = code_and_output(fn, (x, False))
        expected = (x.float() + x.float()).to(x.dtype)
        torch.testing.assert_close(result, expected)

    @skipIfPallas("Pallas gather supports only dim-0 indirect indexing")
    def test_grid_if_reduction_rolling_branch_graph_ids(self):
        """Test for reductions under branch-by-grid control flow.

        Reduction rolling must inspect both _if branch graph ids. The _if node
        also carries branch args, so unpacking the old 3-arg form misses this
        case before codegen.
        """

        # This branch-by-grid pattern is used by the fused kernel work in
        # https://github.com/vllm-project/vllm/pull/38595.
        @helion.kernel(static_shapes=False)
        def fn(
            a: torch.Tensor,
            b: torch.Tensor,
            c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            t = a.size(0)
            ha = hl.specialize(a.shape[1])
            hc = hl.specialize(c.shape[1])
            hmax = hl.specialize(max(a.shape[1], c.shape[1]))
            da = hl.specialize(a.shape[2])
            db = hl.specialize(b.shape[2])
            dc = hl.specialize(c.shape[2])

            out_a = torch.empty_like(a, dtype=torch.float32)
            out_b = torch.empty_like(b, dtype=torch.float32)
            out_c = torch.empty_like(c, dtype=torch.float32)

            for pid, tile_t, tile_h in hl.grid([3, t, hmax]):
                if pid == 0:
                    if tile_h < ha:
                        a_offsets = hl.arange(0, da)
                        a_vals = a[tile_t, tile_h, a_offsets].to(torch.float32)
                        a_scale = torch.rsqrt(
                            torch.sum(a_vals * a_vals, dim=-1) / da + 1.0e-6
                        )
                        out_a[tile_t, tile_h, a_offsets] = a_vals * a_scale
                elif pid == 1:
                    if tile_h < ha:
                        b_offsets = hl.arange(0, db)
                        b_vals = b[tile_t, tile_h, b_offsets].to(torch.float32)
                        b_scale = torch.amax(torch.abs(b_vals), dim=-1)
                        out_b[tile_t, tile_h, b_offsets] = b_vals / b_scale
                elif pid == 2:
                    if tile_h < hc:
                        c_offsets = hl.arange(0, dc)
                        c_vals = c[tile_t, tile_h, c_offsets].to(torch.float32)
                        out_c[tile_t, tile_h, c_offsets] = c_vals + 1.0
            return out_a, out_b, out_c

        t, ha, hc = 4, 8, 12
        da, db, dc = 32, 64, 16
        a = torch.randn(t, ha, da, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(t, ha, db, device=DEVICE, dtype=torch.bfloat16)
        c = torch.randn(t, hc, dc, device=DEVICE, dtype=torch.bfloat16)

        code, result = code_and_output(fn, (a, b, c))
        a_ref = a.float()
        a_scale = torch.rsqrt(
            torch.sum(a_ref * a_ref, dim=-1, keepdim=True) / da + 1.0e-6
        )
        b_ref = b.float()
        b_scale = torch.amax(torch.abs(b_ref), dim=-1, keepdim=True)
        expected = (a_ref * a_scale, b_ref / b_scale, c.float() + 1.0)
        for actual, expected_tensor in zip(result, expected, strict=True):
            torch.testing.assert_close(
                actual, expected_tensor, rtol=1.0e-2, atol=1.0e-2
            )

    @skipIfPallas("Pallas gather supports only dim-0 indirect indexing")
    def test_grid_if_reduction_keepdim_true(self):
        @helion.kernel(static_shapes=False)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x, dtype=torch.float32)
            n = x.size(0)
            d = hl.specialize(x.shape[1])
            for pid, row in hl.grid([2, n]):
                if pid == 0:
                    offsets = hl.arange(0, d)
                    vals = x[row, offsets].to(torch.float32)
                    scale = torch.sum(vals * vals, dim=-1, keepdim=True)
                    out[row, offsets] = vals / scale
            return out

        x = torch.randn(4, 32, device=DEVICE, dtype=torch.bfloat16)
        _, output = code_and_output(fn, (x,))
        expected = x.float() / torch.sum(x.float() * x.float(), dim=-1, keepdim=True)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @skipIfPallas("tensor gather indexing not supported on Pallas")
    def test_optional_tensor_is_none_constexpr(self):
        """Test that `tensor is None` and `tensor is not None` are evaluated as constexpr.

        When an optional tensor parameter is None or not None, the condition should
        be evaluated at compile time and only the relevant branch should be processed.
        """

        @helion.kernel()
        def fn_with_optional(
            data: torch.Tensor,
            optional_indices: torch.Tensor | None,
        ) -> torch.Tensor:
            n = data.size(0)
            out = torch.empty_like(data)
            for tile in hl.tile(n):
                if optional_indices is not None:
                    # Use provided indices
                    idx = optional_indices[tile]
                    out[tile] = data[idx]
                else:
                    # Use identity indices (just copy)
                    out[tile] = data[tile]
            return out

        data = torch.randn(64, device=DEVICE)

        # Test with None - should use else branch
        code_none, result_none = code_and_output(fn_with_optional, (data, None))
        torch.testing.assert_close(result_none, data)
        # Verify that the generated code doesn't have optional_indices at all
        # (it's been eliminated since it's None)
        self.assertNotIn("optional_indices", code_none.split("def fn_with_optional")[0])

        # Test with tensor - should use if branch
        indices = torch.randperm(64, device=DEVICE)
        code_tensor, result_tensor = code_and_output(fn_with_optional, (data, indices))
        expected = data[indices]
        torch.testing.assert_close(result_tensor, expected)
        # Verify that optional_indices IS used in the tensor case
        self.assertIn("optional_indices", code_tensor.split("def fn_with_optional")[0])

    def test_if_not_symbolic_condition(self):
        """``not`` of a tile-dependent condition is traced, not decided at trace time."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0), block_size=8):
                if not (tile_m.begin < 8):
                    out[tile_m, :] = x[tile_m, :] + 1
                if not (tile_m.begin < 16):
                    out[tile_m, :] = out[tile_m, :] * 2
                else:
                    out[tile_m, :] = out[tile_m, :] - 1
                if not tile_m.id:
                    out[tile_m, :] = out[tile_m, :] + 10
                ends_early = not (tile_m.end > 24)
                if not ends_early:
                    out[tile_m, :] = out[tile_m, :] + 100
            return out

        x = torch.randn([32, 16], device=DEVICE)
        expected = torch.zeros_like(x)
        expected[8:] = x[8:] + 1
        expected[16:] *= 2
        expected[:16] -= 1
        expected[:8] += 10
        expected[24:] += 100
        _, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, expected)

    def test_if_not_in_bool_op(self):
        """A traced ``not`` combines with ``and`` instead of folding the branch away."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor, k: int) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0), block_size=8):
                if tile_m.begin >= 8 and tile_m.id != k and not (tile_m.end > 24):
                    out[tile_m, :] = x[tile_m, :] + 1
            return out

        x = torch.randn([32, 16], device=DEVICE)
        for k, rows in ((1, slice(16, 24)), (2, slice(8, 16))):
            expected = torch.zeros_like(x)
            expected[rows] = x[rows] + 1
            _, result = code_and_output(fn, (x, k))
            torch.testing.assert_close(result, expected)

    def test_if_not_loaded_scalar(self):
        """``not`` of a loaded integer scalar is ``== 0``."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0), block_size=8):
                if not flags[tile_m.id]:
                    out[tile_m, :] = x[tile_m, :] + 1
            return out

        x = torch.randn([32, 16], device=DEVICE)
        flags = torch.tensor([1, 0, 2, 0], device=DEVICE, dtype=torch.int32)
        expected = torch.zeros_like(x)
        expected[8:16] = x[8:16] + 1
        expected[24:] = x[24:] + 1
        _, result = code_and_output(fn, (x, flags))
        torch.testing.assert_close(result, expected)

    @xfailIfPallas(
        "Pallas lowers only statically counted while loops (StaticLoopUnroller)"
    )
    def test_while_not_condition(self):
        @helion.kernel(autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                acc = torch.zeros_like(x[tile])
                steps = torch.zeros([], device=x.device, dtype=torch.int32)
                while not (steps >= 4):
                    acc = acc + 1
                    steps = steps + 1
                out[tile] = acc
            return out

        x = torch.zeros(16, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, torch.full_like(x, 4.0))

    @skipIfRefEager("ref eager mode evaluates the conditional expression in Python")
    def test_conditional_expression_not_symbolic_condition_is_rejected(self):
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0), block_size=8):
                scale = 2.0 if not (tile_m.begin < 8) else 3.0
                out[tile_m, :] = x[tile_m, :] * scale
            return out

        x = torch.randn([32, 16], device=DEVICE)
        with self.assertRaisesRegex(
            helion.exc.StatementNotSupported, "dynamic conditional expression"
        ):
            fn.bind((x,))

    @skipIfRefEager("ref eager raises PyTorch's RuntimeError")
    def test_not_of_tile_is_ambiguous(self):
        # As in PyTorch, a tensor of many elements has no single truth value;
        # ``not v`` must not silently become the elementwise ``v == 0``.
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0)):
                v = x[tile_m]
                out[tile_m] = v + (not v)
            return out

        x = torch.randn([64], device=DEVICE)
        with self.assertRaisesRegex(
            helion.exc.TypeInferenceError, "more than one value is ambiguous"
        ):
            fn.bind((x,))

    @skipIfRefEager("checks the device IR")
    def test_if_not_tile_keeps_tensor_predicate(self):
        # ``if not v:`` is ``if v:`` with the branches swapped, so each backend
        # lowers (or rejects) the tile predicate as it does for ``if v:``.
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0)):
                v = x[tile_m]
                if not v:
                    out[tile_m] = v + 1
                else:
                    out[tile_m] = v - 1
            return out

        graphs = fn.bind(
            (torch.randn([64], device=DEVICE),)
        ).host_function.device_ir.graphs
        [if_graph] = [graph for graph in graphs if isinstance(graph, IfGraphInfo)]
        self.assertTrue(if_graph.predicate_is_tensor)
        else_branch = if_graph.else_branch
        assert isinstance(else_branch, int)

        def targets(graph: torch.fx.Graph) -> set[object]:
            return {node.target for node in graph.nodes}

        self.assertIn(torch.ops.aten.sub.Tensor, targets(if_graph.graph))
        self.assertIn(torch.ops.aten.add.Tensor, targets(graphs[else_branch].graph))


@onlyBackends(["triton", "cute"])
class TestUniformBranchBarriers(RefEagerTestBase, TestCase):
    @skipIfRefEager("checks the barrier placement in the generated code")
    @skipIfTileIR("TileIR emits no tl.debug_barrier to order the racing accesses")
    def test_uniform_branch_orders_racing_accesses(self):
        # Inside the branch, the full-row store and the per-column update
        # address ``out`` through different threads.  The condition (a grid
        # tile's begin or an ``hl.grid`` index against a kernel argument, or a
        # flag nothing in the kernel writes) is one value per CTA, so CuTe can
        # order them with a block-wide barrier inside the branch instead of
        # rejecting the kernel.
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def by_tile(x: torch.Tensor, rows: int) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for tile_m in hl.tile(m, block_size=8):
                if tile_m.begin < rows:
                    out[tile_m, :] = x[tile_m, :] + 1
                    for tile_n in hl.tile(n, block_size=8):
                        out[tile_m, tile_n] += x[tile_m, tile_n]
            return out

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def by_grid(x: torch.Tensor, rows: int) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i in hl.grid(m):
                if i < rows:
                    out[i, :] = x[i, :] + 1
                    for tile_n in hl.tile(n, block_size=8):
                        out[i, tile_n] += x[i, tile_n]
            return out

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def by_loaded_flag(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for tile_m in hl.tile(m, block_size=8):
                # Nothing in the kernel can write ``flags`` (only the fresh
                # ``out`` is written): every thread loads the same flag.
                if flags[tile_m.begin] > 0:
                    out[tile_m, :] = x[tile_m, :] + 1
                    for tile_n in hl.tile(n, block_size=8):
                        out[tile_m, tile_n] += x[tile_m, tile_n]
            return out

        x = torch.randn(32, 16, device=DEVICE)
        flags = (torch.arange(32, device=DEVICE) < 16).to(torch.float32)
        expected = torch.zeros_like(x)
        expected[:16] = 2 * x[:16] + 1
        for kernel, args in (
            (by_tile, (x, 16)),
            (by_grid, (x, 16)),
            (by_loaded_flag, (x, flags)),
        ):
            with self.subTest(kernel=kernel.name):
                code, result = code_and_output(kernel, args)
                torch.testing.assert_close(result, expected)
                if _get_backend() == "cute":
                    self.assertTrue(
                        any(
                            "cute.arch.sync_threads()" in ast.unparse(node.body)
                            for node in ast.walk(ast.parse(code))
                            if isinstance(node, ast.If)
                        ),
                        code,
                    )


# Triton and CuTe emit Python ``if``/``for`` statements that assign the
# variables the phi after them merges (Pallas returns branch and loop values
# from ``lax.cond``/``fori_loop`` instead).
@onlyBackends(["triton", "cute"])
class TestControlFlowJoins(RefEagerTestBase, TestCase):
    def test_loop_keeps_fp32_accumulators_as_emitted(self):
        # Only values whose DSL type can drift are joined; an fp32 accumulator
        # of a reduction or matmul loop keeps the loop exactly as emitted.
        @helion.kernel(static_shapes=False)
        def row_sums(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n, block_size=16):
                    acc = acc + x[tile_m, tile_n].to(torch.float32).sum(-1)
                out[tile_m] = acc
            return out

        @helion.kernel(static_shapes=False)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            n = y.size(1)
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k, block_size=16):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(32, 64, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 32, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(row_sums, (x,))
        torch.testing.assert_close(result, x.float().sum(-1), rtol=1e-4, atol=1e-4)
        self.assertNotIn("_cute_join_cast", code)
        code, result = code_and_output(matmul, (x, y))
        torch.testing.assert_close(result, x.float() @ y.float(), rtol=1e-2, atol=1e-2)
        self.assertNotIn("_cute_join_cast", code)

    @skipIfTileIR("TileIR's mul keeps x_bf16 * 0.5 in fp32, changing the carry's type")
    def test_loop_carries_half_precision_value(self):
        # The body's ``value * 0.5`` is Float32 in CuTe DSL; the carried value
        # keeps the type it had before the loop.
        @helion.kernel(static_shapes=False)
        def halve_per_tile(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                value = x[tile_m, 0]
                for _tile_n in hl.tile(n, block_size=16):
                    value = value * 0.5
                out[tile_m] = value
            return out

        @helion.kernel(static_shapes=False)
        def count_tiles(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.int32, device=x.device)
            for tile_m in hl.tile(m):
                index = tile_m.index + 0
                for _tile_n in hl.tile(n, block_size=16):
                    index = index + 1
                out[tile_m] = index
            return out

        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(10, 64, device=DEVICE, dtype=dtype)
            # Four 16-column tiles halve each row's first element exactly.
            _, result = code_and_output(halve_per_tile, (x,))
            torch.testing.assert_close(result, x[:, 0] * 0.0625, rtol=0, atol=0)
        _, result = code_and_output(count_tiles, (x,))
        torch.testing.assert_close(
            result, torch.arange(10, device=DEVICE, dtype=torch.int32) + 4
        )

    def test_if_branch_output_names_outer_value(self):
        # ``tile_n.index`` names a value defined before the if; the other
        # branch's value must not overwrite it for later uses of tile_n.
        @helion.kernel(static_shapes=False)
        def fn(x: torch.Tensor, flag: bool) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros([m, n], dtype=torch.int32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
                if flag:
                    idx = tile_n.index
                else:
                    idx = tile_n.index + 100
                out[tile_m, tile_n] = idx[None, :] + x[tile_m, tile_n]
            return out

        x = torch.zeros(3, 40, device=DEVICE, dtype=torch.int32)
        expected = torch.arange(40, device=DEVICE, dtype=torch.int32).expand(3, 40)
        for flag, offset in ((True, 0), (False, 100)):
            _, result = code_and_output(fn, (x, flag))
            torch.testing.assert_close(result, expected + offset)

    def test_if_branch_reassigns_half_precision_quotient(self):
        # A 16-bit true division stays fp32 on Triton until a consumer casts,
        # while torch.exp of a 16-bit value is computed in fp32 and rounded.
        @helion.kernel(static_shapes=False)
        def fn(x: torch.Tensor, y: torch.Tensor, flag: bool) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                value = x[tile] / y[tile]
                if flag:
                    value = torch.exp(value)
                out[tile] = value
            return out

        x = torch.rand(1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.rand(1000, device=DEVICE, dtype=torch.bfloat16) + 1.0
        for flag in (True, False):
            _, result = code_and_output(fn, (x, y, flag))
            expected = torch.exp(x / y) if flag else x / y
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_if_branch_reassigns_half_precision_value(self):
        @helion.kernel(static_shapes=False)
        def fn(x: torch.Tensor, flag: bool) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                value = x[tile]
                if flag:
                    value = value * 2.0
                out[tile] = value
            return out

        @helion.kernel(static_shapes=False)
        def carried(x: torch.Tensor, flag: bool) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                value = x[tile_m, 0] * 0.5
                for _tile_n in hl.tile(n):
                    if flag:
                        value = value * 0.5
                out[tile_m] = value
            return out

        x = torch.randn(1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(8, 64, device=DEVICE, dtype=torch.bfloat16)
        for flag in (True, False):
            _, result = code_and_output(fn, (x, flag))
            torch.testing.assert_close(result, x * 2.0 if flag else x)
            # One 64-column tile: the branch scales once, exactly.
            _, result = code_and_output(carried, (y, flag), block_sizes=[8, 64])
            torch.testing.assert_close(result, y[:, 0] * (0.25 if flag else 0.5))


if __name__ == "__main__":
    unittest.main()
