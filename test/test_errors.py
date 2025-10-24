from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
import helion.language as hl


@helion.kernel()
def _test_inner_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] * 2
    return out


@helion.kernel()
def _test_outer_kernel_calling_inner(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = _test_inner_kernel(x[tile])
    return out


class TestErrors(RefEagerTestDisabled, TestCase):
    def test_autotune_no_valid_configs(self):
        class FakeKernel:
            def __init__(self) -> None:
                self.settings = helion.Settings(
                    autotune_accuracy_check=False,
                    autotune_precompile=False,
                    autotune_log_level=0,
                )
                from helion.autotuner.config_spec import ConfigSpec

                self.config_spec = ConfigSpec()
                self.configs: list[helion.Config] = []

            def compile_config(self, config: helion.Config, allow_print: bool = False):
                return lambda *args: None

            def format_kernel_decorator(
                self, config: helion.Config, settings: helion.Settings
            ) -> str:
                return "@helion.kernel(...)"

            def to_triton_code(
                self,
                config: helion.Config,
                *,
                emit_repro_caller: bool = False,
                output_origin_lines: bool | None = None,
            ) -> str:
                return ""

        fake_kernel = FakeKernel()
        search = DifferentialEvolutionSearch(fake_kernel, args=())

        def fake_parallel(
            self: PopulationBasedSearch, to_check: list[list[object]]
        ) -> list[PopulationMember]:
            members = []
            for flat_values in to_check:
                cfg = self.config_gen.unflatten(flat_values)
                members.append(
                    PopulationMember(
                        lambda *args: None,
                        [float("inf")],
                        flat_values,
                        cfg,
                    )
                )
            return members

        with (
            mock.patch.object(
                PopulationBasedSearch, "parallel_benchmark_flat", fake_parallel
            ),
            self.assertRaises(helion.exc.NoConfigFound),
        ):
            search.autotune()

    def test_shape_mismatch_missing_keepdims(self):
        """Binary op should detect broadcast shape mismatch from reduction without keep_dims.

        This mirrors the softmax pattern where a row-wise reduction loses the
        dimension and then is subtracted from a 2D tensor without keep_dims.
        """

        # Mirror scratch.py behavior exactly
        @helion.kernel(autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, _tile_n in hl.tile(out.shape):
                amax = torch.amax(x[tile_m, :], dim=1)
                out_rows = torch.exp(x[tile_m, :] - amax)
                out_rows = out_rows / out_rows.sum(dim=1)
                out[tile_m, :] = out_rows
            return out

        with self.assertRaises(helion.exc.ShapeMismatch):
            fn(torch.randn(32, 64, device=DEVICE))

    def test_tile_unpacking(self):
        @helion.kernel()
        def sum_kernel(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = x.new_empty(batch, hidden)
            for tile_batch, tile_hidden in hl.tile(batch, hidden):
                out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
            return out

        with self.assertRaises(helion.exc.FailedToUnpackTile):
            code_and_output(sum_kernel, (torch.randn(2, 3, 4, device=DEVICE),))

    def test_tile_overpacking(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_wrapped_in_tuple in hl.tile([batch]):
                out[tile_wrapped_in_tuple] = x[tile_wrapped_in_tuple, :].sum(1)
            return out

        with self.assertRaises(helion.exc.OverpackedTile):
            code_and_output(fn, (torch.randn(100, 100, device=DEVICE),))

    def test_tile_invalid_range_unpack(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            m = hl.specialize(m)
            d = x.size(2)
            for _tile_m, _tile_d in hl.tile(m, d):
                pass
            return x

        with self.assertRaises(helion.exc.FailedToUnpackTile):
            code_and_output(fn, (torch.randn(192, 4, 128, device=DEVICE),))

    def test_tile_invalid_range_single_dim(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            start = hl.specialize(x.size(0))
            end = x.size(2)
            for _tile_m in hl.tile(start, end):
                pass
            return x

        with self.assertRaisesRegex(
            helion.exc.InvalidTileRange,
            r"begin=192, end=128",
        ):
            code_and_output(fn, (torch.randn(192, 4, 128, device=DEVICE),))

    def test_invalid_config_insufficient_block_sizes(self):
        """Test that InvalidConfig shows helpful message for missing block sizes."""

        @helion.kernel(config=helion.Config(block_sizes=[32, 64]))
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = torch.empty_like(x)
            for tile_batch, tile_seq, tile_hidden in hl.tile([batch, seq_len, hidden]):
                out[tile_batch, tile_seq, tile_hidden] = x[
                    tile_batch, tile_seq, tile_hidden
                ]
            return out

        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            r"Not enough values for config.*expected 3 block sizes.*got 2.*"
            r"Did you forget to specify block sizes for all your hl\.tile\(\) dimensions\?",
        ):
            code_and_output(
                fn,
                (torch.randn(4, 8, 16, device=DEVICE),),
            )

    def test_rank_mismatch_indexing(self):
        """Test that RankMismatch shows tensor shapes in indexing errors."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile([batch]):
                scalar_val = x[tile_batch].sum()  # 1d index for 2d tensor
                out = scalar_val
            return out

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=2, but got ndim=1.*You have too few indices",
        ):
            code_and_output(fn, (torch.randn(4, 8, device=DEVICE),))

    def test_rank_mismatch_indexing_too_many(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            fill = x.new_empty(batch, batch)
            for tile_batch in hl.tile(batch):
                fill = x[tile_batch, tile_batch]  # 2d index for 1d tensor
            return fill

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=1, but got ndim=2.*You have too many indices",
        ):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_invalid_device_for_loop(self):
        """Test that InvalidDeviceForLoop is raised for invalid for loops on device."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile(batch):
                for i in {1: None, 2: None, 3: None}:
                    out[tile_batch] = x[tile_batch] + i
            return out

        with self.assertRaises(helion.exc.InvalidDeviceForLoop):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_return_inside_grid_loop(self):
        """Test that return statement inside hl.grid loop raises proper error."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.grid(batch):
                if x[tile_batch] > 0:
                    return out  # This should not be allowed
                out[tile_batch] = x[tile_batch] * 2
            return out

        with self.assertRaises(helion.exc.NotAllowedOnDevice):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_assign_without_subscript1(self):
        """Test that modifying host variables inside device loops raises proper error."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            result = torch.empty_like(x)
            for tile_batch in hl.tile(batch):
                # shouldn't be able to modify host variables on device
                result = x[tile_batch] * 2
            return result

        with self.assertRaises(helion.exc.CannotModifyHostVariableOnDevice):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_assign_without_subscript2(self):
        """Test that reading device variables from host context raises proper error."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            for tile_batch in hl.tile(batch):
                result = x[tile_batch] * 2
            return result  # shouldn't be able to read device variable here

        with self.assertRaises(helion.exc.CannotReadDeviceVariableOnHost):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_augmented_assign_without_subscript(self):
        """Test that augmented assignment to host variable in device loop raises proper error."""

        @helion.kernel()
        def bad_fn(grad_out: torch.Tensor) -> torch.Tensor:
            m, n = grad_out.shape
            n = hl.specialize(n)
            grad_block = torch.zeros(n, dtype=torch.float32, device=grad_out.device)

            for tile_m in hl.tile(m):
                dy_m = grad_out[tile_m, :].to(torch.float32)
                # Should use `grad_block[:] += ...` instead
                grad_block += torch.sum(dy_m, dim=0)

            return grad_block

        with self.assertRaises(helion.exc.CannotModifyHostVariableOnDevice):
            code_and_output(bad_fn, (torch.randn(4096, 5632, device=DEVICE),))

    def test_device_tensor_subscript(self):
        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            result = torch.empty_like(x)
            for i in hl.tile(batch):
                tmp = x[i] * 2
                tmp[0] = 1  # This should not be allowed
                result[i] = tmp
            return result

        with self.assertRaises(helion.exc.DeviceTensorSubscriptAssignmentNotAllowed):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_boolean_mask_indexing_error(self):
        @helion.kernel()
        def bad_fn(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.shape):
                masked = x[mask]
                out[tile_m, tile_n] = masked.sum()
            return out

        mask = torch.tensor(
            [[True, False], [False, True]], device=DEVICE, dtype=torch.bool
        )
        with self.assertRaises(helion.exc.DataDependentOutputShapeNotSupported):
            code_and_output(
                bad_fn,
                (torch.randn(2, 2, device=DEVICE), mask),
            )

    def test_torch_nonzero_device_error(self):
        @helion.kernel()
        def torch_nonzero_in_device_code(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.shape):
                nz = torch.nonzero(x)  # should error in device context
                out[tile_m, tile_n] = nz.size(0)
            return out

        with self.assertRaises(helion.exc.DataDependentOutputShapeNotSupported):
            code_and_output(
                torch_nonzero_in_device_code, (torch.randn(2, 2, device=DEVICE),)
            )

    def test_torch_chunk_device_error(self):
        """Test that torch.chunk raises error in device loops and suggests hl.split()."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel_with_chunk(q: torch.Tensor) -> torch.Tensor:
            _, _, M, D = q.shape
            D = hl.specialize(D)
            M = hl.specialize(M)
            q = q.reshape(-1, D)
            total_rows = q.shape[0]
            block_m = hl.register_block_size(M)
            result = hl.zeros([total_rows, D])
            for tile_m in hl.tile(total_rows, block_size=block_m):
                acc = hl.zeros([tile_m, D])

                for _tile_n in hl.tile(M, block_size=block_m):
                    acc = torch.stack(torch.chunk(acc, 2, dim=-1), dim=-2).reshape(
                        acc.shape
                    )
                    acc = acc + 0

                result[tile_m, :] = acc

            return result

        with self.assertRaisesRegex(
            helion.exc.UnsupportedSplitOperation,
            r"torch\.chunk is not supported in Helion device loops.*hl\.split\(\)",
        ):
            code_and_output(
                kernel_with_chunk,
                (torch.randn(1, 1, 128, 128, device=DEVICE, dtype=torch.bfloat16),),
            )

    def test_torch_unbind_device_error(self):
        """Test that torch.unbind raises error in device loops and suggests hl.split()."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def kernel_with_unbind(q: torch.Tensor) -> torch.Tensor:
            _, _, M, D = q.shape
            D = hl.specialize(D)
            M = hl.specialize(M)
            q = q.reshape(-1, D)
            total_rows = q.shape[0]
            block_m = hl.register_block_size(M)
            result = hl.zeros([total_rows, D])
            for tile_m in hl.tile(total_rows, block_size=block_m):
                acc = hl.zeros([tile_m, D])

                for _tile_n in hl.tile(M, block_size=block_m):
                    reshaped = acc.reshape(tile_m, 2, D // 2)
                    acc0, acc1 = torch.unbind(reshaped, dim=1)
                    acc = torch.stack((acc0, acc1), dim=1).reshape(tile_m, D)
                    acc = acc + 0

                result[tile_m, :] = acc

            return result

        with self.assertRaisesRegex(
            helion.exc.UnsupportedSplitOperation,
            r"torch\.unbind is not supported in Helion device loops.*hl\.split\(\)",
        ):
            code_and_output(
                kernel_with_unbind,
                (torch.randn(1, 1, 128, 128, device=DEVICE, dtype=torch.bfloat16),),
            )

    def test_closure_fn(self):
        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            def closure_fn():
                pass

            batch = x.size(0)
            result = torch.empty_like(x)
            for i in hl.tile(batch):
                result[i] = x[i] * 2
            return result

        with self.assertRaises(helion.exc.StatementNotSupported):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_direct_scalar_tensor_in_device_context(self):
        """Test that direct scalar tensor usage gives clear error in device code."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + scalar_tensor  # Error: direct scalar usage
            return result

        with self.assertRaises(helion.exc.HostTensorDirectUsage):
            code_and_output(
                bad_fn,
                (torch.randn(4, 4, device=DEVICE), torch.tensor(3.0, device=DEVICE)),
            )

    def test_control_flow_rank_mismatch_variable_name_and_hints(self):
        @helion.kernel()
        def fn(a: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for ti in hl.tile(a.size(0)):
                if ti.index < 1:
                    x = hl.full([ti], 0.0, dtype=a.dtype)
                else:
                    x = hl.full([ti, ti], 0.0, dtype=a.dtype)
                out[ti] = x.sum()
            return a

        with self.assertRaises(
            helion.exc.ControlFlowTensorMismatch,
        ):
            code_and_output(fn, (torch.randn(4, device=DEVICE),))

    def test_too_many_args(self):
        @helion.kernel()
        def kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for i in hl.tile(x.size()):
                result[i] = x[i]
            return result

        with self.assertRaisesRegex(
            TypeError, r"Too many arguments passed to the kernel, expected: 1 got: 2."
        ):
            a = torch.randn(8, device=DEVICE)
            code_and_output(kernel, (a, a))

    def test_kernel_without_device_loop(self):
        @helion.kernel()
        def bf16_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # No hl.tile/hl.grid loops — should raise a friendly error
            return x + y

        with self.assertRaises(helion.exc.NoDeviceLoopsInKernel):
            x = torch.randn(4, 4, device=DEVICE)
            y = torch.randn(4, 4, device=DEVICE)
            code_and_output(bf16_add, (x, y))

    def test_tile_with_tile(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile1 in hl.tile(x.size()):
                for tile2 in hl.tile(tile1):
                    out[tile2] = x[tile2] + 1
            return out

        with self.assertRaises(helion.exc.TileOfTile):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_nested_kernel_calls(self):
        with self.assertRaisesRegex(
            helion.exc.NestedKernelCallsNotSupported,
            r"Calling a Helion kernel from within another Helion kernel is not supported",
        ):
            code_and_output(
                _test_outer_kernel_calling_inner, (torch.randn(8, device=DEVICE),)
            )

    def test_hl_dot_batch_dim_mismatch(self):
        """Test that hl.dot raises error when batch dimensions don't match."""

        @helion.kernel()
        def kernel_with_dot_mismatch(
            q: torch.Tensor,
            k: torch.Tensor,
        ) -> torch.Tensor:
            m = q.size(0)
            m = hl.specialize(m)
            n = k.size(0)
            n = hl.specialize(n)
            d = q.size(2)

            out = torch.zeros_like(q)
            kT = k.transpose(1, 2)

            for tile_m, tile_d in hl.tile([m, d]):
                q_blk = q[tile_m, :, tile_d]  # [tile_m, H, tile_d]

                for tile_n in hl.tile(n):
                    k_blk = kT[tile_n, tile_d, :]  # [tile_n, tile_d, H]
                    # This will fail: `q_blk` has batch dim `tile_m`, `k_blk` has batch dim `tile_n`
                    qk = hl.dot(q_blk, k_blk)

                out[tile_m, :, tile_d] = qk

            return out

        q = torch.randn(128, 3, 64, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(128, 3, 64, dtype=torch.bfloat16, device=DEVICE)

        with self.assertRaisesRegex(
            helion.exc.DotBatchDimensionMismatch,
            r"got \(tile_m \(symbol: u0\)\) from LHS tensor vs\. \(tile_n \(symbol: u3\)\) from RHS tensor",
        ):
            code_and_output(kernel_with_dot_mismatch, (q, k))


if __name__ == "__main__":
    unittest.main()
