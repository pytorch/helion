from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from typing import Sequence

    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin


class StackTensor(NamedTuple):
    """
    This class should not be instantiated directly. It is the result of hl.stacktensor_like(...).
    It presents a batch of tensors of the same properties (shape, dtype and stride)
    but reside at different memory locations virtually stacked together.

    StackTensor provides a way to perform parallel memory accesses to multiple tensors with a single subscription.


    **Core Concept:**

    Instead of performing separate memory operations on each tensor individually,
    StackTensor allows you to broadcast a single memory operation (hl.load, hl.store, hl.atomic_add,
    hl.signal, hl.wait etc.) to multiple tensor buffers in parallel. This is particularly useful
    for batch processing scenarios where the same operation needs to be applied to multiple tensors.

    **Memory Operation Behavior:**

    - **Loads**: When you index into a StackTensor (e.g., `stack_tensor[i]`),
      it performs the same indexing operation on all underlying tensor buffers and
      returns a new tensor where the results are stacked according to the shape of dev_ptrs.
    - **Stores**: When you assign to a StackTensor (e.g., `stack_tensor[i] = value`),
      the value tensor is "unstacked" - each slice of the value tensor is written to the respective
      underlying tensor buffer. This is the reverse operation of loading.
      (e.g. value[j] is writtent to tensor_j[i]).

    **Shape Semantics:**

    The StackTensor's shape is `dev_ptrs.shape + tensor_like.shape`, where:

    - `dev_ptrs.shape` becomes the stacking dimensions
    - `tensor_like.shape` represents the shape of each individual tensor

    """

    tensor_like: torch.Tensor
    """
    A template host tensor that defines the shape, dtype, and other properties
                    for all tensors in the stack group.
                    Must be a Host tensor (created outside of the device loop).
    """

    dev_ptrs: torch.Tensor
    """
    A tensor containing device pointers (memory buffer addresses) to the actual
                 tensors in device memory.
                 Must be of dtype torch.uint64.
    """

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

    def new_empty(
        self, *args: Sequence[int | torch.SymInt], **kwargs: dict
    ) -> torch.Tensor:
        return self.tensor_like.new_empty(*args, **kwargs)  # pyright: ignore[reportCallIssue]

    # TODO(joydddd): Implement this to support StackTensor in ref mode.
    # def as_tuple_of_tensor(self) -> tuple[torch.Tensor, ...]:
    """
    Returns a tuple of tensors that represent the underlying buffers of the stack tensor.

    This function is useful when you need to access the underlying tensors directly,
    for example, to run in eager mode.

    """


def stacktensor_like(
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
) -> StackTensor:
    """
    Creates a StackTensor from a tensor of data pointers (dev_ptrs) pointing to tensors alike
    residing at different memory locations.

    This function creates a StackTensor that allows you to broadcast memory operations
    to multiple tensor buffers in parallel.

    Must be called inside a helion kernel with dev_ptrs as a device tensor and tensor_like
    as a host tensor.

    Args:
        tensor_like: A template host tensor that defines the shape, dtype, and other properties
                    that each buffer in the stack group should have. Must be a host tensor.
        dev_ptrs: A tensor containing device pointers (memory addresses) to data buffers.
                 Must be of dtype torch.uint64 and must be a device tensor.

    Examples:
        **Basic Load Operation:**

        .. code-block:: python

            @helion.kernel
            def stack_load(dev_ptrs: torch.Tensor, example: torch.Tensor):
                for tile in hl.tile(example.size(0)):
                    ptr_tile = dev_ptrs[:]  # Shape: [num_tensors]
                    stack_tensor = hl.stack_like(example, ptr_tile)
                    # Load from all tensors simultaneously
                    data = stack_tensor[tile]  # Shape: [num_tensors, tile_size]
                return data

        **Store Operation:**

        .. code-block:: python

            @helion.kernel
            def stack_store(
                dev_ptrs: torch.Tensor, example: torch.Tensor, values: torch.Tensor
            ):
                ptr_tile = dev_ptrs[:]  # Shape: [num_tensors]
                stack_tensor = hl.stack_like(example, ptr_tile)

                # Store values of shape [num_tensors, N] to all tensors in parallel
                stack_tensor[:] = values  # slice values[i, :] goes to tensor i

        **Usage Setup:**

        .. code-block:: python

            # Create list of tensors to process
            tensor_list = [torch.randn(16, device="cuda") for _ in range(4)]
            tensor_ptrs = torch.as_tensor(
                [p.data_ptr() for p in tensor_list], dtype=torch.uint64, device="cuda"
            )
            result = stack_load(tensor_ptrs, tensor_list[0])

    Returns:
        A StackTensor object that broadcasts memory operations to all data buffers
        pointed to by dev_ptrs.
    """
    raise exc.NotInsideKernel


@_decorators.device_func_replacement(stacktensor_like)
@_decorators.device_func_replacement(StackTensor)
@_decorators.api(is_device_only=False, allow_host_tensor=True)
def _stack_tensor(
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
) -> StackTensor:
    raise exc.NotInsideKernel


@_decorators.type_propagation(_stack_tensor)
def _(tensor_like: TypeInfo, dev_ptrs: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.type_propagation import StackTensorType
    from .._compiler.type_propagation import TensorType

    assert isinstance(dev_ptrs, TensorType)
    assert isinstance(tensor_like, TensorType)
    if origin.is_host():
        raise exc.StackTensorcOnHost
    if dev_ptrs.origin.is_host():
        raise exc.StackTensorDevPtrOnHost
    if tensor_like.origin.is_device():
        raise exc.StackTensorExampleOnDevice
    if dev_ptrs.fake_value.dtype != torch.uint64:
        raise exc.StackTensorDevPtrDtype(dev_ptrs.fake_value.dtype)
    element_types = {
        "dev_ptrs": dev_ptrs,
        "tensor_like": tensor_like,
    }

    return StackTensorType(origin, element_types)  # pyright: ignore[reportArgumentType]


@_decorators.register_to_device_ir(_stack_tensor)
def _(tracer: object, tensor_like: torch.Tensor, dev_ptrs: torch.Tensor) -> StackTensor:
    return StackTensor(tensor_like, dev_ptrs)
