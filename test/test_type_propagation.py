from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfXPU
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_str()


@onlyBackends(["triton", "cute"])
class TestTypePropagation(RefEagerTestDisabled, TestCase):
    def test_add(self):
        output = type_propagation_report(
            basic_kernels.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_torch_ops_pointwise(self):
        output = type_propagation_report(
            basic_kernels.torch_ops_pointwise,
            torch.ones([1024], dtype=torch.int32),
            torch.ones([1024], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_all_ast_nodes(self):
        output = type_propagation_report(
            import_path(datadir / "all_ast_nodes.py").all_ast_nodes,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
            ignore=True,
        )
        self.assertExpectedJournal(output)

    def test_hl_zeros_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_zeros_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_hl_full_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_full_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_pointwise_device_loop(self):
        output = type_propagation_report(
            basic_kernels.pointwise_device_loop,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_method_call(self):
        @helion.kernel
        def fn(x):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile].sin()
            return out

        output = type_propagation_report(
            fn,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedJournal(output)

    def test_matmul(self):
        output = type_propagation_report(
            import_path(examples_dir / "matmul.py").matmul,
            torch.ones([512, 512]),
            torch.ones([512, 512]),
        )
        self.assertExpectedJournal(output)

    @skipIfXPU("CUDA-only")
    def test_cuda_device_properties(self):
        @helion.kernel
        def use_device_properties(x: torch.Tensor) -> torch.Tensor:
            device = x.device
            props = torch.cuda.get_device_properties(device)
            sm_count = props.multi_processor_count

            n = x.shape[0]
            out = torch.zeros_like(x)

            for worker_id in hl.grid(sm_count):
                for i in hl.grid(n):
                    idx = worker_id + i * sm_count
                    if idx < n:
                        out[idx] = x[idx]
            return out

        x = torch.ones([128], device="cuda")  # @ignore-device-lint
        output = type_propagation_report(use_device_properties, x)
        self.assertExpectedJournal(output)

    def test_symbolic_comparison_preserves_expression(self):
        """A comparison/`and` on symbolic block sizes must propagate the real
        relation (e.g. ``block_m < 64``), not a fresh unbacked bool that discards
        the operands. Preserving the expression keeps the block-size dependency
        so the condition stays resolvable per-config at codegen."""

        @helion.kernel
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            block_m = hl.register_block_size(m)
            block_n = hl.register_block_size(n)
            cmp = block_m < 64  # noqa: F841
            both = block_m < 64 and block_n >= 64  # noqa: F841
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n], block_size=[block_m, block_n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        output = type_propagation_report(fn, torch.ones([128, 128], device=DEVICE))
        # The comparison and the `and` propagate as real relationals over the
        # block-size symbols, e.g. `SymBoolType(u0 < 64)` and a conjunction that
        # still mentions `>= 64`. Before the fix they became opaque unbacked
        # bools like `SymBoolType(Eq(u2, 1))` that dropped the operands.
        self.assertIn("SymBoolType(", output)
        self.assertIn("< 64)", output)
        self.assertIn(">= 64", output)
        self.assertNotRegex(output, r"SymBoolType\(Eq\(u\d+, 1\)\)")

    @skipIfXPU("CUDA-only")
    def test_cuda_device_properties_unsupported_attribute(self):
        @helion.kernel
        def use_unsupported_property(x: torch.Tensor) -> torch.Tensor:
            device = x.device
            props = torch.cuda.get_device_properties(device)
            for i in hl.grid(x.shape[0]):
                unsupported = props.total_memory  # attribute not supported yet
                x[i] = unsupported
            return x

        x = torch.ones([16], device="cuda")  # @ignore-device-lint
        with self.assertRaisesRegex(
            exc.TypeInferenceError,
            r"Attribute 'total_memory' is not supported on .*test_type_propagation.py",
        ):
            type_propagation_report(use_unsupported_property, x)

    def test_and_between_optional_tensors(self):
        @helion.kernel()
        def kernel(
            t: torch.Tensor,
            c: torch.Tensor | None = None,
            d: torch.Tensor | None = None,
        ):
            a = torch.empty_like(t)
            for h in hl.tile(a.size(0)):
                if c is not None and d is not None:
                    a[h] = t[h] + c[h] + d[h]
                else:
                    a[h] = t[h]
            return a

        x = torch.ones([16], device=DEVICE)
        output = type_propagation_report(kernel, x)
        self.assertExpectedJournal(output)

    @skipIfXPU("CUDA-only")
    def test_list_iteration(self):
        @helion.kernel()
        def kernel_list_iteration(
            tensor_list: list[torch.Tensor],
        ) -> torch.Tensor:
            (M,) = tensor_list[0].shape
            result = torch.zeros_like(tensor_list[0])
            for tile_m in hl.tile(M):
                acc = hl.zeros([tile_m], dtype=torch.float32)
                for tensor in tensor_list:
                    acc = acc + tensor[tile_m]
                result[tile_m] = acc
            return result

        size = 16
        tensors = [
            torch.ones(size, device=DEVICE, dtype=torch.float32) * (i + 1)
            for i in range(4)
        ]
        output = type_propagation_report(kernel_list_iteration, tensors)
        self.assertExpectedJournal(output)


if __name__ == "__main__":
    unittest.main()
