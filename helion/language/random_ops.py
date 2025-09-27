from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["rand"]


@_decorators.api(tiles_as_sizes=True)
def rand(
    shape: list[object],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    The main propose of ``hl.rand`` is to explicitly pass a seed arg for deterministic
    randomness in helion kernels, whereas ``torch.rand_like`` doesn't take seed arg
    (though it can seeded globally)`. ``hl.rand`` lower to ``tl.rand(seed, offset)`` with ``offset``
    built from a linear range over the allocation and reshaped to the given shape.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For host allocations, use ``torch.rand()``.

    Args:
        shape: A list of sizes
        seed: int seed for the random number generator
        dtype: currently only float32 supported

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with random values

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(x)
                (m,) = x.shape
                for (tile_m,) in hl.tile([m]):
                    output[tile_m] = hl.rand([tile_m], seed=seed)
                return output

    """
    raise NotInsideKernel


@_decorators.register_fake(rand)
def _rand_fake(
    shape: list[int | torch.SymInt],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.codegen(rand)
def _rand_codegen(state: CodegenState) -> ast.AST:
    """Generate tl.rand() code with tile-specific offsets for unique RNG per tile."""
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)

    tensor_shape = fake_value.size()
    shape_str = state.device_function.tile_strategy.shape_str(tensor_shape)
    seed_ast = state.ast_arg(1)
    env = CompileEnvironment.current()

    # 1D case directly to reuse precomputed indices
    if len(tensor_shape) == 1:
        block_id = env.get_block_id(tensor_shape[0])
        if block_id is None:
            raise RuntimeError(
                "hl.rand() requires tiled dimensions. "
                "Use hl.rand() inside hl.tile() loops with tile variables."
            )
        index_var = state.codegen.index_var(block_id)
        offs_expr = f"{index_var}.reshape({shape_str})"
    else:
        # for multi dim cases compute linear offset using row-major order
        offset_components = []

        for dim_idx, size in enumerate(tensor_shape):
            block_id = env.get_block_id(size)
            if block_id is not None:
                offset_var = state.codegen.offset_var(block_id)

                # calculate stride for this dimension (product of remaining dimensions)
                if dim_idx < len(tensor_shape) - 1:
                    stride_components = []
                    for next_dim_idx in range(dim_idx + 1, len(tensor_shape)):
                        next_size = tensor_shape[next_dim_idx]
                        next_block_id = env.get_block_id(next_size)

                        if next_block_id is not None:
                            stride_components.append(f"_BLOCK_SIZE_{next_block_id}")
                        else:
                            stride_components.append(str(next_size))

                    stride = " * ".join(stride_components)
                    offset_components.append(f"({offset_var}) * ({stride})")
                else:
                    # for last dimension no stride multiplication needed
                    offset_components.append(offset_var)

        if not offset_components:
            raise RuntimeError(
                "hl.rand() requires tiled dimensions. "
                "Use hl.rand() inside hl.tile() loops with tile variables."
            )

        numel = " * ".join(shape_str.strip("[]").split(","))
        base_offset = " + ".join(offset_components)
        offs_expr = f"({base_offset} + tl.arange(0, {numel})).reshape({shape_str})"

    expr = f"tl.rand({{seed}}, {offs_expr})"
    return expr_from_string(expr, seed=seed_ast)


@_decorators.get_masked_value(rand)
def _(
    node: torch.fx.Node,
) -> float:
    return 0


@_decorators.ref(rand)
def _(
    shape: list[int | RefTile],
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    processed_shape: list[int] = []
    for s in shape:
        if isinstance(s, RefTile):
            processed_shape.append(s.end - s.begin)
        else:
            processed_shape.append(int(s))
    env = CompileEnvironment.current()
    gen = torch.Generator(device=env.device if device is None else device)
    gen.manual_seed(seed)
    return torch.rand(
        processed_shape,
        dtype=dtype,
        generator=gen,
        device=env.device if device is None else device,
    )
