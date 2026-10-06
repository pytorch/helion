from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton", "cute"])
class TestStackTensor(RefEagerTestDisabled, TestCase):
    def test_stack_load_across_explicit_barrier(self):
        @helion.kernel
        def stack_load_kernel(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            m = hl.specialize(dev_ptrs.size(0))
            n = example_tensor.size(0)
            tmp = torch.empty(
                [m, n], dtype=example_tensor.dtype, device=dev_ptrs.device
            )
            out = torch.empty_like(tmp)

            for tile in hl.tile(n):
                tensors = hl.stacktensor_like(example_tensor, dev_ptrs[:])
                tmp[:, tile] = tensors[tile]
            hl.barrier()
            for tile in hl.tile(n):
                out[:, tile] = tmp[:, tile] + 1
            return out

        tensor_list = [
            torch.randn(4, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
        ]
        tensor_ptrs = torch.as_tensor(
            [tensor.data_ptr() for tensor in tensor_list],
            device=DEVICE,
            dtype=torch.uint64,
        )
        _code, result = code_and_output(
            stack_load_kernel,
            (tensor_ptrs, tensor_list[0]),
            block_sizes=[4, 4],
            pid_type="persistent_blocked",
        )

        torch.testing.assert_close(result, torch.stack(tensor_list) + 1)

    def test_stack_load_grid(self):
        @helion.kernel
        def stack_load_kernel(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            M = hl.specialize(dev_ptrs.size(0))
            N = example_tensor.size(0)
            out = torch.empty(M, N, dtype=torch.bfloat16, device=dev_ptrs.device)

            for i in hl.grid(N):
                ptr_tile = dev_ptrs[:]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                out[:, i] = tensors[i]
            return out

        tensor_list = [
            torch.randn(4, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )
        code, result = code_and_output(stack_load_kernel, (tensor_ptrs, tensor_list[0]))
        torch.testing.assert_close(result, torch.stack(tensor_list))

    def test_stack_load_2d_tensors(self):
        @helion.kernel
        def stack_load_kernel(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            M = dev_ptrs.size(0)
            N1, N2 = example_tensor.size()
            out = torch.empty(M, N1, N2, dtype=torch.bfloat16, device=dev_ptrs.device)

            for tile1, tile2 in hl.tile([N1, N2]):
                ptr_tile = dev_ptrs[:]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                out[:, tile1, tile2] = tensors[tile1, tile2]
            return out

        tensor_list = [
            torch.randn(4, 4, device=DEVICE, dtype=torch.bfloat16) for _ in range(8)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )

        code, result = code_and_output(
            stack_load_kernel, (tensor_ptrs, tensor_list[0]), block_size=[4, 4]
        )
        torch.testing.assert_close(result, torch.stack(tensor_list))

    def test_stack_load_2d_dev_ptrs(self):
        @helion.kernel
        def stack_load_kernel_2d(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            M1, M2 = dev_ptrs.size()
            N = example_tensor.size(0)
            out = torch.empty(M1, M2, N, dtype=torch.bfloat16, device=dev_ptrs.device)

            for tile in hl.tile(N, block_size=4):
                ptr_tile = dev_ptrs[:, :]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                out[:, :, tile] = tensors[tile]
            return out

        tensor_list = [
            torch.randn(4, device=DEVICE, dtype=torch.bfloat16) for _ in range(16)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        ).reshape(4, 4)

        code_batched, result = code_and_output(
            stack_load_kernel_2d, (tensor_ptrs, tensor_list[0])
        )
        torch.testing.assert_close(result, torch.stack(tensor_list).reshape(4, 4, -1))

        @helion.kernel
        def stack_load_2d_looped(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            M1, M2 = dev_ptrs.size()
            N = example_tensor.size(0)
            out = torch.empty(M1, M2, N, dtype=torch.bfloat16, device=dev_ptrs.device)

            for tile in hl.tile(N, block_size=4):
                for i in range(M1):
                    ptr_tile = dev_ptrs[i, :]
                    tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                    out[i, :, tile] = tensors[tile]
            return out

        code_looped, result = code_and_output(
            stack_load_2d_looped, (tensor_ptrs, tensor_list[0])
        )
        torch.testing.assert_close(result, torch.stack(tensor_list).reshape(4, 4, -1))

    def test_stack_mask(self):
        @helion.kernel
        def stack_load_w_mask(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> torch.Tensor:
            M = dev_ptrs.size(0)
            N = example_tensor.size(0)
            out = torch.empty(M, N, dtype=torch.bfloat16, device=dev_ptrs.device)

            for tile in hl.tile(N, block_size=4):
                for stack_tile in hl.tile(M, block_size=4):
                    ptr_tile = dev_ptrs[stack_tile]
                    tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                    out[:, tile] = tensors[tile]
            return out

        tensor_list = [
            torch.randn(15, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )

        code, result = code_and_output(stack_load_w_mask, (tensor_ptrs, tensor_list[0]))
        torch.testing.assert_close(result, torch.stack(tensor_list))

    def test_stack_store_grid(self):
        @helion.kernel
        def stack_store_kernel(
            x: torch.Tensor,
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> None:
            N = x.size(0)
            hl.specialize(dev_ptrs.size(0))

            for i in hl.grid(N):
                ptr_tile = dev_ptrs[:]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                tensors[i] = x[None, i]

        tensor_list = [
            torch.empty(16, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )

        x = torch.randn(16, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            stack_store_kernel, (x, tensor_ptrs, tensor_list[0])
        )

        for tensor in tensor_list:
            torch.testing.assert_close(tensor, x)

    def test_stack_store_broadcast_masked(self):
        @helion.kernel
        def stack_store_kernel(
            x: torch.Tensor,
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> None:
            N = x.size(0)
            hl.specialize(dev_ptrs.size(0))

            for tile in hl.tile(N, block_size=4):
                ptr_tile = dev_ptrs[:]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                x_tile = x[tile]
                tensors[tile] = x_tile[None, :]

        tensor_list = [
            torch.empty(15, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )

        x = torch.randn(15, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            stack_store_kernel, (x, tensor_ptrs, tensor_list[0])
        )

        for tensor in tensor_list:
            torch.testing.assert_close(tensor, x)

    def test_stack_store_scatter(self):
        @helion.kernel
        def stack_store_arange_kernel(
            dev_ptrs: torch.Tensor,
            example_tensor: torch.Tensor,
        ) -> None:
            N = example_tensor.size(0)
            M = hl.specialize(dev_ptrs.size(0))

            for i in hl.grid(N):
                ptr_tile = dev_ptrs[:]
                tensors = hl.stacktensor_like(example_tensor, ptr_tile)
                x = hl.arange(M)
                tensors[i] = x

        tensor_list = [
            torch.empty(15, device=DEVICE, dtype=torch.int32) for _ in range(4)
        ]
        tensor_ptrs = torch.as_tensor(
            [p.data_ptr() for p in tensor_list], device=DEVICE, dtype=torch.uint64
        )

        code, result = code_and_output(
            stack_store_arange_kernel, (tensor_ptrs, tensor_list[0])
        )

        for i, tensor in enumerate(tensor_list):
            assert tensor.eq(i).all().item()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _stack_memory_proof_kernel(x, dev_ptrs, write: hl.constexpr):
    count = hl.specialize(dev_ptrs.size(0))
    out = torch.empty((count, x.size(0)), dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0), block_size=4):
        tensors = hl.stacktensor_like(x, dev_ptrs[:])
        if write:
            tensors[tile] = x[tile][None, :]
            out[:, tile] = x[tile][None, :]
        else:
            out[:, tile] = tensors[tile]
    return out


class TestStackTensorOwnershipCPU(TestCase):
    def test_indirect_memory_keeps_load_ownership_conservative(self):
        from unittest.mock import patch

        from torch.fx import Node

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        from helion._compiler.cute.local_atomic import _reachable_graphs
        from helion._compiler.cute.register_loads import host_load_is_readonly
        from helion._compiler.cute.register_loads import lane_private_load
        from helion.language import memory_ops

        with (
            _mock_cuda_unavailable(),
            _target(),
            _forbid_native_compile(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        ):
            for width in (5, 16):
                for write in (False, True):
                    with self.subTest(width=width, write=write):
                        x = torch.arange(width).float()
                        pointers = torch.zeros(3, dtype=torch.uint64)
                        bound = _cpu_bind(
                            _stack_memory_proof_kernel, (x, pointers, write)
                        )
                        code = bound.to_code(bound.config_spec.default_config())
                        self.assertTrue(code)
                        host = bound.host_function
                        graphs = _reachable_graphs(host.device_ir.graphs)
                        loads = [
                            node
                            for graph in graphs
                            for node in graph.graph.nodes
                            if node.target is memory_ops.load
                        ]
                        checked = 0
                        with bound.env, host:
                            for load in loads:
                                if write or not isinstance(load.args[0], Node):
                                    # A disjoint pointer array says nothing about
                                    # its pointees. Indirect writes can alias x.
                                    self.assertFalse(
                                        host_load_is_readonly(load, bound.env, graphs)
                                    )
                                    if not isinstance(load.args[0], Node):
                                        self.assertFalse(
                                            lane_private_load(load, bound.env)
                                        )
                                    checked += 1
                        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
