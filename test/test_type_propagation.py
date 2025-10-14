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
from helion._testing import skipIfXPU
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_str()


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


if __name__ == "__main__":
    unittest.main()
