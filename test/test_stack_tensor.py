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


@onlyBackends(["triton"])
class TestStackTensor(RefEagerTestDisabled, TestCase):
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
        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code_batched + code_looped)

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
        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)

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

        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
