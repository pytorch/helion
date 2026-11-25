"""Does a kernel touch memory past the end of one of its tensors?

A read past a tensor is silent while the memory behind it is mapped and an
illegal memory access (Xid 31, a dead CUDA context) when it is not, which
depends on what the allocator did before. ``tensor_at_block_end`` makes the
question deterministic and fault-free. The tensor lives in CUDA managed
memory that is resident on the CPU, and its last byte is the last byte of a
2 MiB UVM block. A GPU access to a managed page migrates the page, which
removes its CPU mapping, so ``/proc/self/pagemap`` afterwards names the pages
the kernel touched; UVM's own read-ahead stays inside the faulting block, so
any page gone from the next block was touched by the kernel. The pages past
the tensor are part of the same allocation: the stray access migrates a page
instead of faulting, whether or not the kernel is correct.

Everything the probe relies on is checked, and reported as a skip rather than
a pass: the driver bindings, the managed allocation, a readable pagemap, the
CPU residency before a launch, and (``PageTouches.observable``) that the
kernel's own reads of the tensor did migrate its pages.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import dataclasses
import importlib
from typing import TYPE_CHECKING
from unittest import SkipTest

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

PAGE_BYTES = 4096
# UVM manages, and reads ahead within, 2 MiB blocks of virtual addresses.
UVM_BLOCK_BYTES = 2 << 20
_PAGEMAP_PRESENT = np.uint64(1 << 63)
_BITS = {size: np.dtype(f"<i{size}") for size in (1, 2, 4, 8)}


@dataclasses.dataclass(frozen=True)
class PageTouches:
    """Pages of a ``BlockEndTensor`` allocation the GPU touched since ``reset``."""

    tensor_pages: int
    past_end_pages: int
    # Offset from the tensor's end of the end of the last touched page (0: none).
    past_end_reach: int

    @property
    def observable(self) -> bool:
        """The kernel's reads of the tensor itself were seen."""
        return self.tensor_pages > 0


class BlockEndTensor:
    """A CUDA tensor in managed memory that ends where a UVM block ends."""

    def __init__(self, shape: Sequence[int], dtype: torch.dtype) -> None:
        try:
            driver = importlib.import_module("cuda.bindings.driver")
        except ImportError as error:
            raise SkipTest("needs cuda.bindings") from error
        self._driver = driver
        self.shape = tuple(int(extent) for extent in shape)
        # NumPy has no bfloat16: the allocation is exposed as same-width
        # integers and viewed as ``dtype`` on the torch side.
        self._bits = _BITS[torch.empty(0, dtype=dtype).element_size()]
        self.nbytes = int(np.prod(self.shape)) * self._bits.itemsize
        assert self.nbytes > 0
        # The driver call below needs PyTorch's context current on this thread.
        torch.zeros(1, device="cuda")
        blocks = -(-self.nbytes // UVM_BLOCK_BYTES)
        # The tensor's blocks, one of slack to align in and one behind it.
        self._size = (blocks + 2) * UVM_BLOCK_BYTES
        error, pointer = driver.cuMemAllocManaged(
            self._size, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value
        )
        if error != driver.CUresult.CUDA_SUCCESS:
            raise SkipTest(f"no CUDA managed memory: {error}")
        self._region = int(pointer)
        aligned = -(-self._region // UVM_BLOCK_BYTES) * UVM_BLOCK_BYTES
        self.end = aligned + blocks * UVM_BLOCK_BYTES
        self.base = self.end - self.nbytes
        assert self._region <= self.base
        assert self.end + UVM_BLOCK_BYTES <= self._region + self._size
        self._bytes = np.ctypeslib.as_array(
            (ctypes.c_uint8 * self._size).from_address(self._region)
        )
        self._host = np.ctypeslib.as_array(
            (ctypes.c_uint8 * self.nbytes).from_address(self.base)
        )
        self._host = self._host.view(self._bits).reshape(self.shape)
        self.__cuda_array_interface__ = {
            "shape": self.shape,
            "typestr": self._bits.str,
            "data": (self.base, False),
            "version": 3,
            "strides": None,
        }
        self._bytes[:] = 0
        try:
            self.tensor = torch.as_tensor(self, device="cuda").view(dtype)
            assert self.tensor.data_ptr() == self.base
            self.reset()
        except BaseException:
            self._free()
            raise

    def write(self, values: torch.Tensor) -> None:
        """Copy a CPU tensor in from the CPU side, then ``reset``.

        A device-side copy would itself migrate the pages it writes.
        """
        assert values.device.type == "cpu" and values.dtype == self.tensor.dtype
        assert tuple(values.shape) == self.shape
        bits = torch.from_numpy(np.empty(0, dtype=self._bits)).dtype
        self._host[...] = values.contiguous().view(bits).numpy()
        self.reset()

    def _present(self) -> np.ndarray:
        pages = self._size // PAGE_BYTES
        try:
            with open("/proc/self/pagemap", "rb") as pagemap:
                pagemap.seek(self._region // PAGE_BYTES * 8)
                entries = np.frombuffer(pagemap.read(pages * 8), dtype=np.uint64)
        except OSError as error:
            raise SkipTest(f"cannot read /proc/self/pagemap: {error}") from error
        if len(entries) != pages:
            raise SkipTest("short read of /proc/self/pagemap")
        return (entries & _PAGEMAP_PRESENT) != 0

    def reset(self) -> None:
        """Bring every page back to the CPU: the start of an observation."""
        torch.cuda.synchronize()
        # One CPU write per page migrates it back without changing its bytes.
        self._bytes[::PAGE_BYTES] += 0
        if not self._present().all():
            raise SkipTest("managed pages do not show as CPU resident")

    def touches(self) -> PageTouches:
        """What the GPU touched since ``reset`` (synchronizes the device)."""
        torch.cuda.synchronize()
        gone = np.flatnonzero(~self._present()) * PAGE_BYTES + self._region
        first = self.base // PAGE_BYTES * PAGE_BYTES
        past = gone[gone >= self.end]
        return PageTouches(
            tensor_pages=int(((gone >= first) & (gone < self.end)).sum()),
            past_end_pages=len(past),
            past_end_reach=int(past.max()) + PAGE_BYTES - self.end if len(past) else 0,
        )

    def _free(self) -> None:
        (error,) = self._driver.cuMemFree(self._region)
        assert error == self._driver.CUresult.CUDA_SUCCESS, error

    def close(self) -> None:
        torch.cuda.synchronize()
        del self.tensor, self._host, self._bytes
        self._free()


@contextmanager
def tensor_at_block_end(
    shape: Sequence[int], dtype: torch.dtype = torch.float32
) -> Iterator[BlockEndTensor]:
    """A zero-filled CUDA tensor of ``shape`` followed by observable pages.

    ``.write(values)`` fills it from the CPU, ``.reset()`` starts an
    observation and ``.touches()`` ends it, around the launch in between.
    Raises ``unittest.SkipTest`` when page touches cannot be observed here.
    """
    probe = BlockEndTensor(shape, dtype)
    try:
        yield probe
    finally:
        probe.close()
