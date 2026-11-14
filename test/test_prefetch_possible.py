"""``tensor_prefetch_possible``: which ``hl.prefetch`` calls can take the
TMA box form, on mocked compile environments (no GPU)."""

from __future__ import annotations

import operator
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest
import torch

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.triton import memory_ops
from helion._testing import TestCase
from helion._testing import onlyBackends

pytestmark = pytest.mark.cpu_only


@onlyBackends(["triton"])
class TestTensorPrefetchPossible(TestCase):
    """``tensor_prefetch_possible``, the config-independent half of the tensor
    form (test_prefetch checks the config's box through generated code)."""

    def _possible(
        self,
        tensor: torch.Tensor,
        index: list[object],
        stride_known: bool = True,
        extra_mask: bool = False,
        unkeyed: bool = False,
        allocator: str = "native",
    ) -> bool:
        """On an sm_100 target with static sizes (and strides, unless
        ``stride_known`` is False), of a tensor whose base the bound kernel
        can key (unless ``unkeyed``), under the CUDA caching allocator
        ``allocator``."""
        env = SimpleNamespace(
            ptx_capability=(10, 0),
            static_kernel_tensor_extent=lambda n: n,
            known_stride=lambda t, d: t.stride(d) if stride_known else None,
            known_equal=operator.eq,
            get_block_id=lambda _: None,
            tensor_descriptor_alignment_source=lambda t: None if unkeyed else t,
        )
        with (
            mock.patch.object(CompileEnvironment, "current", return_value=env),
            mock.patch.object(
                memory_ops, "supports_tensor_descriptor", return_value=True
            ),
            mock.patch.object(
                torch.cuda.memory, "get_allocator_backend", return_value=allocator
            ),
        ):
            return memory_ops.tensor_prefetch_possible(tensor, index, extra_mask)

    def test_tma_limits(self) -> None:
        """Rows that are no multiple of 16 bytes, strides only the first call
        has, partial slices, an extra mask, an unkeyed base and a full last-dim
        slice no row of boxes covers keep the pointer form."""
        rows = slice(None)
        bf16 = torch.bfloat16
        tensor = torch.empty(16, 64, 256, dtype=bf16)
        self.assertTrue(self._possible(tensor, [0, rows, rows]))
        self.assertTrue(
            self._possible(torch.empty(64, 1, 16 * 256, dtype=bf16), [0, rows, rows])
        )
        storage = torch.empty(16 * 64 * 257, dtype=bf16)
        self.assertFalse(
            self._possible(
                storage.as_strided([16, 64, 256], [64 * 257, 257, 1]), [0, rows, rows]
            )
        )
        self.assertFalse(self._possible(tensor, [0, rows, rows], stride_known=False))
        self.assertFalse(self._possible(tensor, [0, slice(0, 32), rows]))
        self.assertFalse(self._possible(tensor, [0, rows, rows], extra_mask=True))
        # No input base to key (a tensor the host code allocates).
        self.assertFalse(self._possible(tensor, [0, rows, rows], unkeyed=True))
        # A full last-dim slice over more than 16 boxes, or of rows under 16
        # bytes, takes the box form under no config.
        wide = torch.empty(4, 17 * 256, dtype=bf16)
        self.assertTrue(self._possible(wide[:, : 16 * 256], [0, rows]))
        self.assertFalse(self._possible(wide, [0, rows]))
        narrow = torch.empty(64 * 8, dtype=bf16).as_strided([64, 4], [8, 1])
        self.assertFalse(self._possible(narrow, [0, rows]))

    def test_cuda_malloc_async_keeps_the_pointer_form(self) -> None:
        """The maps' memory comes from a private pool, which cudaMallocAsync
        does not have: under that allocator the prefetch keeps the pointer
        form."""
        tensor = torch.empty(16, 64, 256, dtype=torch.bfloat16)
        index = [0, slice(None), slice(None)]
        self.assertTrue(self._possible(tensor, index))
        self.assertFalse(self._possible(tensor, index, allocator="cudaMallocAsync"))


if __name__ == "__main__":
    unittest.main()
