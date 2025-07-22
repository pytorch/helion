from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin


class MulticastTensor(NamedTuple):
    """
    MulticastTensor is a batch of tensors of the same properties (shape, dtype and stride)
    but reside at different memory locations.
    It provides a way to multicast memory accesses to multiple tensors with a single subscription.

    **Core Concept:**
    Instead of performing separate memory operations on each tensor individually,
    MulticastTensor allows you to "multicast" a single memory operation (hl.load, hl.store, hl.atomic_add,
    hl.signal, hl.wait etc.) to multiple tensor buffers simultaneously. This is particularly useful
    for batch processing scenarios where the same operation needs to be applied to multiple tensors.

    **Memory Operation Behavior:**
    - **Loads**: When you index into a MulticastTensor (e.g., `multicast_tensor[i]`),
      it performs the same indexing operation on all underlying tensor buffers and
      returns a new tensor where the results are stacked according to the shape of dev_ptrs.
    - **Stores**: When you assign to a MulticastTensor (e.g., `multicast_tensor[i] = value`),
      the value tensor is "unstacked" - each slice of the value tensor is written to the respective
      underlying tensor buffer. This is the reverse operation of loading.
      (e.g. value[j] is writtent to tensor_j[i]).

    **Shape Semantics:**
    The MulticastTensor's shape is `dev_ptrs.shape + tensor_like.shape`, where:
    - `dev_ptrs.shape` represents the "batch" dimensions (how many tensors are being multicasted)
    - `tensor_like.shape` represents the shape of each individual tensor


    Attributes:
        tensor_like: A template host tensor that defines the shape, dtype, and other properties
                    that all tensors in the multicast group should have.
        dev_ptrs: A tensor containing device pointers (memory buffer addresses) to the actual
                 tensors in device memory. Must be of dtype torch.uint64.

    Properties:
        dtype: The data type of the tensors in the multicast group. Inherited from tensor_like.
        shape: The shape of the multicasted tensor. Computed as dev_ptrs.shape + tensor_like.shape.
    """

    tensor_like: torch.Tensor
    dev_ptrs: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor_like.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor_like.device

    @property
    def shape(self) -> torch.Size:
        return self.dev_ptrs.shape + self.tensor_like.shape

    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        index: list[object] | torch.Tensor,
    ) -> torch.Tensor:
        raise exc.NotInsideKernel

    def __setitem__(  # pyright ignore[reportIncompatibleMethodOverride]
        self,
        index: list[object] | torch.Tensor,
        value: torch.Tensor | bool | float,
    ) -> None:
        raise exc.NotInsideKernel


def multicast_like(
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
) -> MulticastTensor:
    """
    Creates a MulticastTensor from a tensor of data pointers (dev_ptrs) pointing to tensors alike
    residing at different memory locations.

    This function creates a MulticastTensor that allows you to multicast the memory operations
    on multiple tensor buffers simultaneously.

    Must be called inside a helion kernel with dev_ptrs as a device tensor and tensor_like
    as a host tensor.

    Args:
        tensor_like: A template host tensor that defines the shape, dtype, and other properties
                    that each buffer in the multicast group should have. Must be a host tensor.
        dev_ptrs: A tensor containing device pointers (memory addresses) to data buffers.
                 Must be of dtype torch.uint64 and must be a device tensor.

    Examples:
        **Basic Load Operation:**

        .. code-block:: python

            @helion.kernel
            def multicast_load(dev_ptrs: torch.Tensor, example: torch.Tensor):
                for tile in hl.tile(example.size(0)):
                    ptr_tile = dev_ptrs[:]  # Shape: [num_tensors]
                    multicast_tensor = hl.multicast_like(example, ptr_tile)
                    # Load from all tensors simultaneously
                    data = multicast_tensor[tile]  # Shape: [num_tensors, tile_size]
                return data

        **Store Operation:**

        .. code-block:: python

            @helion.kernel
            def multicast_store(
                dev_ptrs: torch.Tensor, example: torch.Tensor, values: torch.Tensor
            ):
                ptr_tile = dev_ptrs[:]  # Shape: [num_tensors]
                multicast_tensor = hl.multicast_like(example, ptr_tile)
                # Store to all tensors simultaneously - values must have multicast dimension
                # values.shape should be [num_tensors, ...] to match multicast_tensor.shape
                multicast_tensor[:] = values  # Each values[i] goes to tensor i

        **Usage Setup:**

        .. code-block:: python

            # Create list of tensors to process
            tensor_list = [torch.randn(16, device="cuda") for _ in range(4)]
            tensor_ptrs = torch.as_tensor(
                [p.data_ptr() for p in tensor_list], dtype=torch.uint64, device="cuda"
            )
            result = multicast_load(tensor_ptrs, tensor_list[0])

    Returns:
        A MulticastTensor object that multicasts memory operations to all data buffers
        pointed to by dev_ptrs.
    """
    raise exc.NotInsideKernel


@_decorators.device_func_replacement(multicast_like)
@_decorators.device_func_replacement(MulticastTensor)
@_decorators.api(is_device_only=False, allow_host_tensor=True)
def _multicast(
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
) -> MulticastTensor:
    raise exc.NotInsideKernel


@_decorators.type_propagation(_multicast)
def _(tensor_like: TypeInfo, dev_ptrs: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.type_propagation import MulticastTensorType
    from .._compiler.type_propagation import TensorType

    assert isinstance(dev_ptrs, TensorType)
    assert isinstance(tensor_like, TensorType)
    if origin.is_host():
        raise exc.MulticastTensorcOnHost
    if dev_ptrs.origin.is_host():
        raise exc.MulticastTensorDevPtrOnHost
    if tensor_like.origin.is_device():
        raise exc.MulticastTensorExampleOnDevice
    if dev_ptrs.fake_value.dtype != torch.uint64:
        raise exc.MulticastTensorDevPtrDtype(dev_ptrs.fake_value.dtype)
    element_types = {
        "dev_ptrs": dev_ptrs,
        "tensor_like": tensor_like,
    }

    return MulticastTensorType(origin, element_types)  # pyright: ignore[reportArgumentType]


@_decorators.register_to_device_ir(_multicast)
def _(
    tracer: object, tensor_like: torch.Tensor, dev_ptrs: torch.Tensor
) -> MulticastTensor:
    return MulticastTensor(tensor_like, dev_ptrs)
