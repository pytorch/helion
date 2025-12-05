from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.indexing_strategy import StackIndexingStrategy
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["rand", "randint"]


@_decorators.api(tiles_as_sizes=True)
def rand(
    shape: list[object],
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    hl.rand provides a Philox-based pseudorandom number generator (PRNG) that operates independently of PyTorch’s global random seed.
    Instead, it requires an explicit seed argument. Offsets are derived from the full logical sizes of the tiles specified in the shape argument.

    Args:
        shape: A list of sizes for the output tensor
        seed: A single element int64 tensor or int literal
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of float32 dtype filled with uniform random values in [0, 1)

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(x)
                (m,) = x.shape
                for tile_m in hl.tile(m):
                    output[tile_m] = hl.rand([tile_m], seed=42)
                return output

    """
    raise NotInsideKernel


@_decorators.register_fake(rand)
def _rand_fake(
    shape: list[int | torch.SymInt],
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=torch.float32,
        device=env.device if device is None else device,
    )


@_decorators.codegen(rand, "triton")
def _rand_codegen(state: CodegenState) -> ast.AST:
    """
    Generate tl.rand() code with global indices for deterministic RNG per element.

    This implementation uses improved dimension detection and broadcasting logic
    while maintaining compatibility with the existing approach.
    """
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)

    env = CompileEnvironment.current()
    tensor_shape = fake_value.size()
    ndim = len(tensor_shape)
    if ndim == 0:
        raise ValueError("hl.rand() requires at least one dimension")

    seed_ast = state.ast_arg(1)

    index_vars = []
    size_names: list[str] = []
    for i in range(ndim):
        size = tensor_shape[i]
        block_id = env.get_block_id(size)
        if block_id is not None:
            index_vars.append(state.codegen.index_var(block_id))
            original_tensor_size = env.block_sizes[block_id].size
            assert isinstance(original_tensor_size, torch.SymInt), (
                f"Expected SymInt, got {type(original_tensor_size)}"
            )
            size_names.append(
                state.device_function.sympy_expr(original_tensor_size._sympy_())
            )
        else:
            rdim = env.allocate_reduction_dimension(size)
            index_vars.append(state.codegen.index_var(rdim.block_id))
            assert isinstance(rdim.var, torch.SymInt), (
                f"Expected SymInt, got {type(rdim.var)}"
            )
            size_names.append(state.device_function.sympy_expr(rdim.var._sympy_()))

    if ndim == 1:
        offset_expr = expr_from_string(index_vars[0])
    else:
        offset_parts: list[str] = []
        for i in range(ndim):
            broadcast_slice = StackIndexingStrategy.get_element_broadcast_slice(i, ndim)
            broadcasted_index = f"{index_vars[i]}{broadcast_slice}"
            if i < ndim - 1:
                # pyrefly: ignore [no-matching-overload]
                stride_expr = " * ".join(map("({})".format, size_names[i + 1 :]))
                offset_parts.append(f"{broadcasted_index} * {stride_expr}")
            else:
                offset_parts.append(broadcasted_index)
        offset_expr = expr_from_string(" + ".join(offset_parts))
    return expr_from_string(
        "tl.rand({seed}, {offset})", seed=seed_ast, offset=offset_expr
    )


@_decorators.get_masked_value(rand)
def _(
    node: torch.fx.Node,
) -> float:
    return 0


@_decorators.ref(rand)
def _(
    shape: list[int | RefTile],
    seed: int | torch.Tensor,
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
    if isinstance(seed, torch.Tensor):
        gen.manual_seed(int(seed.item()))
    else:
        gen.manual_seed(seed)
    return torch.rand(
        processed_shape,
        dtype=torch.float32,
        generator=gen,
        device=env.device if device is None else device,
    )


@_decorators.api(tiles_as_sizes=True)
def randint(
    shape: list[object],
    low: int,
    high: int,
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    hl.randint provides a Philox-based pseudorandom integer generator (PRNG) that operates independently of PyTorch's global random seed.
    Instead, it requires an explicit seed argument. Offsets are derived from the full logical sizes of the tiles specified in the shape argument.

    Args:
        shape: A list of sizes for the output tensor
        low: Lowest integer to be drawn from the distribution (inclusive)
        high: One above the highest integer to be drawn from the distribution (exclusive)
        seed: A single element int64 tensor or int literal
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of int32 dtype filled with random integers in [low, high)

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
                (m,) = x.shape
                for tile_m in hl.tile(m):
                    output[tile_m] = hl.randint([tile_m], low=0, high=10, seed=42)
                return output

    """
    raise NotInsideKernel


@_decorators.register_fake(randint)
def _randint_fake(
    shape: list[int | torch.SymInt],
    low: int,
    high: int,
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=torch.int32,
        device=env.device if device is None else device,
    )


@_decorators.codegen(randint, "triton")
def _randint_codegen(state: CodegenState) -> ast.AST:
    """
    Generate tl.randint() code with global indices for deterministic RNG per element.

    This implementation generates random int32 values and applies modulo arithmetic
    to produce values in the range [low, high).
    """
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)

    env = CompileEnvironment.current()
    tensor_shape = fake_value.size()
    ndim = len(tensor_shape)
    if ndim == 0:
        raise ValueError("hl.randint() requires at least one dimension")

    # Get low, high, and seed from arguments
    low_ast = state.ast_arg(1)
    high_ast = state.ast_arg(2)
    seed_ast = state.ast_arg(3)

    index_vars = []
    size_names: list[str] = []
    for i in range(ndim):
        size = tensor_shape[i]
        block_id = env.get_block_id(size)
        if block_id is not None:
            index_vars.append(state.codegen.index_var(block_id))
            original_tensor_size = env.block_sizes[block_id].size
            assert isinstance(original_tensor_size, torch.SymInt), (
                f"Expected SymInt, got {type(original_tensor_size)}"
            )
            size_names.append(
                state.device_function.sympy_expr(original_tensor_size._sympy_())
            )
        else:
            rdim = env.allocate_reduction_dimension(size)
            index_vars.append(state.codegen.index_var(rdim.block_id))
            assert isinstance(rdim.var, torch.SymInt), (
                f"Expected SymInt, got {type(rdim.var)}"
            )
            size_names.append(state.device_function.sympy_expr(rdim.var._sympy_()))

    if ndim == 1:
        offset_expr = expr_from_string(index_vars[0])
    else:
        offset_parts: list[str] = []
        for i in range(ndim):
            broadcast_slice = StackIndexingStrategy.get_element_broadcast_slice(i, ndim)
            broadcasted_index = f"{index_vars[i]}{broadcast_slice}"
            if i < ndim - 1:
                # pyrefly: ignore [no-matching-overload]
                stride_expr = " * ".join(map("({})".format, size_names[i + 1 :]))
                offset_parts.append(f"{broadcasted_index} * {stride_expr}")
            else:
                offset_parts.append(broadcasted_index)
        offset_expr = expr_from_string(" + ".join(offset_parts))

    # Generate: low + (tl.randint(seed, offset).to(tl.int32) % (high - low))
    # Cast to int32 first to handle negative low values properly
    # This ensures values are in [low, high) range
    return expr_from_string(
        "{low} + tl.abs(tl.randint({seed}, {offset}).to(tl.int32)) % ({high} - {low})",
        low=low_ast,
        high=high_ast,
        seed=seed_ast,
        offset=offset_expr,
    )


@_decorators.get_masked_value(randint)
def _(
    node: torch.fx.Node,
) -> int:
    return 0


@_decorators.ref(randint)
def _(
    shape: list[int | RefTile],
    low: int,
    high: int,
    seed: int | torch.Tensor,
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
    if isinstance(seed, torch.Tensor):
        gen.manual_seed(int(seed.item()))
    else:
        gen.manual_seed(seed)
    return torch.randint(
        low,
        high,
        processed_shape,
        dtype=torch.int32,
        generator=gen,
        device=env.device if device is None else device,
    )
