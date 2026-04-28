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
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_str()


@onlyBackends(["triton"])
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
            r"Attribute 'total_memory' is not supported on",
        ):
            type_propagation_report(use_unsupported_property, x)

    @skipIfXPU("CUDA-only")
    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_device_properties_arithmetic(self):
        """Regression test for https://github.com/pytorch/helion/issues/778:
        'error when operating on python int' when doing math on a value from
        torch.cuda.get_device_properties().

        The crash requires: (1) multi_processor_count assigned then used in
        arithmetic that produces a compound sympy expression (e.g. 4*u0),
        and (2) that expression multiplied with another symbol in device code
        so sympy decomposes back to the original free symbol u0 whose
        expr_to_origin still points at SourceOrigin.
        """

        @helion.kernel(config=helion.Config(block_sizes=[1, 8192]))
        def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            n = hl.specialize(n)
            out = torch.empty_like(x)
            m_block = hl.register_block_size(m)
            n_block = hl.register_block_size(n)
            m_count = (m + m_block - 1) // m_block
            n_count = (n + n_block - 1) // n_block
            num_workers = torch.cuda.get_device_properties(
                x.device
            ).multi_processor_count
            num_workers = num_workers + num_workers + num_workers + num_workers
            total_work = m_count * n_count
            work_per_worker = (total_work + num_workers - 1) // num_workers
            for worker_id in hl.grid(num_workers):
                for work_item in hl.grid(work_per_worker):
                    work_n = (worker_id + num_workers * work_item) % n_count
                    work_m = (worker_id + num_workers * work_item) // n_count
                    work_n_start = work_n * n_block
                    work_n_end = min(work_n_start + n_block, n)
                    work_m_start = work_m * m_block
                    work_m_end = min(work_m_start + m_block, m)
                    for tile_n, tile_m in hl.tile(
                        [work_n_start, work_m_start],
                        [work_n_end, work_m_end],
                        block_size=[n_block, m_block],
                    ):
                        x_tile = x[tile_m, tile_n].to(torch.float32)
                        out[tile_m, tile_n] = x_tile.to(out.dtype)
            return out

        x = torch.randn(1000, 1024, device=DEVICE)
        w = torch.randn(1024, device=DEVICE)
        output = type_propagation_report(fn, x, w)
        self.assertExpectedJournal(output)
        code, result = code_and_output(fn, (x, w))
        torch.testing.assert_close(result, x)
        self.assertIn("num_workers", code)
        # The Triton kernel should receive num_workers as a parameter,
        # not repeat the get_device_properties() call in device code.
        kernel_code = code.split("@triton.jit")[1].split("\ndef ")[0]
        self.assertNotIn("get_device_properties", kernel_code)

    @skipIfXPU("CUDA-only")
    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_device_properties_doubled(self):
        """Variant of test_device_properties_arithmetic using 2*u0 instead of 4*u0."""

        @helion.kernel(config=helion.Config(block_sizes=[1, 8192]))
        def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            n = hl.specialize(n)
            out = torch.empty_like(x)
            m_block = hl.register_block_size(m)
            n_block = hl.register_block_size(n)
            m_count = (m + m_block - 1) // m_block
            n_count = (n + n_block - 1) // n_block
            num_workers = torch.cuda.get_device_properties(
                x.device
            ).multi_processor_count
            num_workers = num_workers + num_workers
            total_work = m_count * n_count
            work_per_worker = (total_work + num_workers - 1) // num_workers
            for worker_id in hl.grid(num_workers):
                for work_item in hl.grid(work_per_worker):
                    work_n = (worker_id + num_workers * work_item) % n_count
                    work_m = (worker_id + num_workers * work_item) // n_count
                    work_n_start = work_n * n_block
                    work_n_end = min(work_n_start + n_block, n)
                    work_m_start = work_m * m_block
                    work_m_end = min(work_m_start + m_block, m)
                    for tile_n, tile_m in hl.tile(
                        [work_n_start, work_m_start],
                        [work_n_end, work_m_end],
                        block_size=[n_block, m_block],
                    ):
                        x_tile = x[tile_m, tile_n].to(torch.float32)
                        out[tile_m, tile_n] = x_tile.to(out.dtype)
            return out

        x = torch.randn(1000, 1024, device=DEVICE)
        w = torch.randn(1024, device=DEVICE)
        output = type_propagation_report(fn, x, w)
        self.assertExpectedJournal(output)
        code, result = code_and_output(fn, (x, w))
        torch.testing.assert_close(result, x)
        self.assertIn("num_workers", code)
        kernel_code = code.split("@triton.jit")[1].split("\ndef ")[0]
        self.assertNotIn("get_device_properties", kernel_code)

    @skipIfXPU("CUDA-only")
    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_device_properties_division(self):
        """Device-code division that strips the compound factor, leaving bare u0.

        num_workers = sm_count + sm_count  (= 2*u0, overwrites sm_count)
        In device code: num_workers // 2  simplifies to u0, which has no
        named variable and only a SourceOrigin.
        """

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            num_workers = torch.cuda.get_device_properties(
                x.device
            ).multi_processor_count
            # Overwrite so u0 has no named variable at grid boundary
            num_workers = num_workers + num_workers  # 2*u0
            for tile in hl.tile(n):
                # num_workers // 2 simplifies to bare u0 in sympy,
                # exercising _resolve_via_compound in codegen
                out[tile] = x[tile] + (num_workers // 2)
            return out

        x = torch.randn(1024, device="cuda")  # @ignore-device-lint
        output = type_propagation_report(fn, x)
        self.assertExpectedJournal(output)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + sm_count)
        self.assertIn("num_workers", code)

    @skipIfXPU("CUDA-only")
    @skipIfRefEager("Config tests not applicable in ref eager mode")
    def test_device_properties_division_as_grid(self):
        """Bare u0 used as hl.grid size (host-side HostFunction.sympy_expr)."""

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.zeros_like(x)
            num_workers = torch.cuda.get_device_properties(
                x.device
            ).multi_processor_count
            num_workers = num_workers + num_workers  # 2*u0 (overwrites)
            # num_workers // 2 = u0 used as grid size
            for worker_id in hl.grid(num_workers // 2):
                for tile in hl.tile(n):
                    out[tile] = x[tile]
            return out

        x = torch.randn(1024, device="cuda")  # @ignore-device-lint
        output = type_propagation_report(fn, x)
        self.assertExpectedJournal(output)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x)

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
