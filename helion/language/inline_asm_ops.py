from __future__ import annotations

from typing import Sequence
from typing import overload

import torch

from .. import exc
from . import _decorators

__all__ = ["inline_asm_elementwise"]


@overload
@_decorators.api(is_device_only=True)
def inline_asm_elementwise(
    asm: str,
    constraints: str,
    args: Sequence[torch.Tensor],
    dtype: torch.dtype,
    is_pure: bool,
    pack: int,
) -> torch.Tensor: ...


@overload
@_decorators.api(is_device_only=True)
def inline_asm_elementwise(
    asm: str,
    constraints: str,
    args: Sequence[torch.Tensor],
    dtype: Sequence[torch.dtype],
    is_pure: bool,
    pack: int,
) -> tuple[torch.Tensor, ...]: ...


@_decorators.api(is_device_only=True)
def inline_asm_elementwise(
    asm: str,
    constraints: str,
    args: Sequence[torch.Tensor],
    dtype: torch.dtype | Sequence[torch.dtype],
    is_pure: bool,
    pack: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Execute inline assembly over a tensor. Essentially, this is map
    where the function is inline assembly.

    The input tensors args are implicitly broadcasted to the same shape.
    dtype can be a tuple of types, in which case the output is a
    tuple of tensors.

    Each invocation of the inline asm processes pack elements at a
    time. Exactly which set of inputs a block receives is unspecified.
    Input elements of size less than 4 bytes are packed into 4-byte
    registers.

    This op does not support empty dtype -- the inline asm must
    return at least one tensor, even if you don't need it. You can work
    around this by returning a dummy tensor of arbitrary type; it shouldn't
    cost you anything if you don't use it.

    Args:
        asm: assembly to run. Must match target's assembly format.
        constraints: asm constraints in LLVM format
        args: the input tensors, whose values are passed to the asm block
        dtype: the element type(s) of the returned tensor(s)
        is_pure: if true, the compiler assumes the asm block has no side-effects
        pack: the number of elements to be processed by one instance of inline assembly

    Returns:
        one tensor or a tuple of tensors of the given dtypes
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(inline_asm_elementwise)
def _(
    asm: str,
    constraints: str,
    args: Sequence[torch.Tensor],
    dtype: torch.dtype | Sequence[torch.dtype],
    is_pure: bool,
    pack: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    from .._compiler.compile_environment import CompileEnvironment

    # Basic validation
    if not isinstance(asm, str):
        raise exc.InvalidAPIUsage(f"asm must be a string, got {type(asm)}")
    if not isinstance(constraints, str):
        raise exc.InvalidAPIUsage(
            f"constraints must be a string, got {type(constraints)}"
        )
    if not isinstance(is_pure, bool):
        raise exc.InvalidAPIUsage(f"is_pure must be a bool, got {type(is_pure)}")
    if not isinstance(pack, int):
        raise exc.InvalidAPIUsage(f"pack must be an int, got {type(pack)}")

    # Determine if we have multiple outputs
    if isinstance(dtype, (tuple, list)):
        dtypes = list(dtype)
        has_multiple_outputs = True
    else:
        dtypes = [dtype]
        has_multiple_outputs = False

    # Validate dtype(s)
    for dt in dtypes:
        if not isinstance(dt, torch.dtype):
            raise exc.InvalidAPIUsage(f"dtype must be torch.dtype, got {type(dt)}")

    # Broadcast all inputs to the same shape
    if args:
        broadcast_shape = args[0].shape
        for arg in args[1:]:
            if arg.shape != broadcast_shape:
                broadcast_shape = torch.broadcast_shapes(broadcast_shape, arg.shape)
    else:
        # For empty args, we need to infer the shape from context
        # The problem is that without input tensors, we can't determine the proper broadcast shape
        # However, when used in a tile context, the output should match the tile shape
        # For the fake function, we'll use a simple placeholder shape that the compiler will handle
        broadcast_shape = (1,)

    env = CompileEnvironment.current()
    if has_multiple_outputs:
        results = []
        for dt in dtypes:
            # Type assertion: dt is guaranteed to be torch.dtype due to validation above
            assert isinstance(dt, torch.dtype)
            result = torch.empty(broadcast_shape, dtype=dt, device=env.device)
            results.append(result)
        return tuple(results)

    # Type assertion: dtypes[0] is guaranteed to be torch.dtype due to validation above
    assert isinstance(dtypes[0], torch.dtype)
    return torch.empty(broadcast_shape, dtype=dtypes[0], device=env.device)
