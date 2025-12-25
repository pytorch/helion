from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect
from torch.fx.experimental import proxy_tensor

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["load", "store"]

# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}

@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor / stack tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel

@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | tuple,
    list[object],
    torch.Tensor | torch.SymInt | float | int,
    torch.Tensor | None,
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes(index)

    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, value, extra_mask)

    if isinstance(tensor, torch.Tensor):
        return (tensor, index, value, extra_mask)

    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")

@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None

@_decorators.register_to_device_ir(store)
def _(
    tracer: "proxy_tensor.PythonKeyTracer",
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Handle store during tracing, applying epilogue if one is registered.

    This uses the HOP-based approach for epilogue fusion:
    1. Get the current store index
    2. Check if there's an epilogue callable for this store
    3. If so, call the epilogue to transform the value, then store
    4. The proxy mode captures the epilogue operations as part of tracing
    5. At codegen time, the epilogue is inlined into the store
    """
    from .._compiler.fusion.convert import get_epilogue_tracing_state

    state = get_epilogue_tracing_state()

    if state is not None and isinstance(tensor, torch.Tensor):
        store_idx = state.get_and_increment_store_idx()
        buffer_name = state.store_map.get(store_idx)

        # Check for epilogue callable
        epilogue_fn = state.get_epilogue_callable(store_idx)

        if epilogue_fn is not None:
            # Call the epilogue function directly during re-tracing.
            # Re-enable proxy mode so the epilogue operations are captured.
            proxy_mode = _decorators.get_disabled_proxy_mode()

            if proxy_mode is not None:
                with proxy_mode:
                    transformed_value = epilogue_fn(value, index)
            else:
                transformed_value = epilogue_fn(value, index)

            # Now store the transformed value
            # Create the store proxy node
            proxy_out = tracer.create_proxy(
                "call_function",
                store,
                *_decorators.args_to_proxies(tracer, (tensor, index, transformed_value, extra_mask), {}),
            )
            proxy_tensor.track_tensor_tree(None, proxy_out, constant=None, tracer=tracer)

            # Track original value for multi-output epilogue
            if buffer_name:
                state.stored_values[buffer_name] = value

            return None

        # Track value for multi-output epilogue
        if buffer_name:
            state.stored_values[buffer_name] = value
        # Check for multi-output epilogue
        is_multi_output_store = (
            state.multi_output_op_fn is not None and
            len(state.multi_output_acc_names) >= 2 and
            buffer_name in state.multi_output_acc_names
        )

        if is_multi_output_store:
            proxy_mode = _decorators.get_disabled_proxy_mode()

            if not state.is_last_store(store_idx):
                # Intermediate store - store normally and remember tensor
                if 'first_output_tensor' not in state.__dict__:
                    state.first_output_tensor = tensor
                    state.first_output_index = index

                # Create normal store
                proxy_out = tracer.create_proxy(
                    "call_function",
                    store,
                    *_decorators.args_to_proxies(tracer, (tensor, index, value, extra_mask), {}),
                )
                proxy_tensor.track_tensor_tree(None, proxy_out, constant=None, tracer=tracer)
                return None

            # Last store - apply multi-output epilogue
            acc_name_0 = state.multi_output_acc_names[0]
            acc_name_1 = state.multi_output_acc_names[1]

            if acc_name_0 in state.stored_values and acc_name_1 in state.stored_values:
                val0 = state.stored_values[acc_name_0]
                val1 = state.stored_values[acc_name_1]
                op_fn = state.multi_output_op_fn

                # Apply the operation with proxy mode enabled
                if proxy_mode is not None:
                    with proxy_mode:
                        value = op_fn(val0, val1)
                else:
                    value = op_fn(val0, val1)

                # Use first output tensor
                if hasattr(state, 'first_output_tensor') and state.first_output_tensor is not None:
                    tensor = state.first_output_tensor
                    index = state.first_output_index
                elif state.multi_output_epilogue_buffer is not None:
                    tensor = state.multi_output_epilogue_buffer

    # Create the FX proxy for store with (possibly transformed) value
    proxy_out = tracer.create_proxy(
        "call_function",
        store,
        *_decorators.args_to_proxies(tracer, (tensor, index, value, extra_mask), {}),
    )
    proxy_tensor.track_tensor_tree(None, proxy_out, constant=None, tracer=tracer)

    return None

@_decorators.codegen(store, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor, subscript, value, extra_mask = state.proxy_arg(0), state.proxy_arg(1), state.ast_arg(2), state.ast_args[3]
    assert isinstance(subscript, (list, tuple)) and isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_store_index += 1
        device_fn.device_memory_op_index += 1

        return device_fn.get_indexing_strategy(indexing_idx).codegen_store(state, tensor, [*subscript], value, extra_mask)
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")



# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    from .ref_tile import RefTile

    # Normalize indices and identify tensor indices
    indices = []
    tensor_idx_positions = []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index
        # pyrefly: ignore [bad-argument-type]
        indices.append(idx)
        if isinstance(idx, torch.Tensor):
            tensor_idx_positions.append(i)

    # Handle broadcasting for multiple tensor indices
    if len(tensor_idx_positions) > 1:
        grids = torch.meshgrid(
            # pyrefly: ignore [bad-argument-type]
            *(indices[i] for i in tensor_idx_positions),
            indexing="ij",
        )
        for i, grid in zip(tensor_idx_positions, grids, strict=False):
            # pyrefly: ignore [unsupported-operation]
            indices[i] = grid

    if extra_mask is not None:
        mask = extra_mask.to(torch.bool)

        # Check bounds for tensor indices
        for i, idx in enumerate(indices):
            if isinstance(idx, torch.Tensor):
                mask = mask & (idx >= 0) & (idx < tensor.shape[i])
        mask_count = int(mask.sum().item())
        if mask_count == 0:
            return

        # Use index_put_ for masked stores
        valid_indices = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                valid_indices.append(idx[mask].long())
            else:
                idx_val = int(idx) if isinstance(idx, torch.SymInt) else idx
                valid_indices.append(
                    # pyrefly: ignore [no-matching-overload]
                    torch.full(
                        (mask_count,), idx_val, dtype=torch.long, device=tensor.device
                    )
                )

        if isinstance(value, torch.Tensor):
            values = value[mask]
        else:
            val = int(value) if isinstance(value, torch.SymInt) else value
            values = torch.full(
                (mask_count,), val, dtype=tensor.dtype, device=tensor.device
            )

        # Check for duplicate indices - this is undefined behavior in Triton
        if valid_indices:
            stacked = torch.stack(valid_indices, dim=1)
            unique_count = stacked.unique(dim=0).size(0)
            if unique_count < stacked.size(0):
                raise exc.DuplicateStoreIndicesError(
                    "hl.store with duplicate indices has undefined behavior in compiled mode. "
                    "The order in which values are written to the same memory location is "
                    "non-deterministic and may vary between Triton versions and backends."
                )

        tensor.index_put_(tuple(valid_indices), values, accumulate=False)
        return

    # Simple assignment
    tensor[tuple(indices)] = int(value) if isinstance(value, torch.SymInt) else value

@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range. It also accepts an optional
    `eviction_policy` which is forwarded to the underlying Triton `tl.load`
    call to control the cache eviction behavior (e.g., "evict_last").

    Args:
        tensor: The tensor / stack tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
        eviction_policy: Optional Triton load eviction policy to hint cache behavior
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel

@_decorators.prepare_args(load)
def _(
    tensor: torch.Tensor | StackTensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> tuple[torch.Tensor | tuple, list[object], torch.Tensor | None, str | None]:
    from .tile_proxy import Tile

    index = Tile._tiles_to_sizes(index)
    if isinstance(tensor, StackTensor):
        return (tuple(tensor), index, extra_mask, eviction_policy)
    assert isinstance(tensor, torch.Tensor)
    return (tensor, index, extra_mask, eviction_policy)

@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        target_shape = SubscriptIndexing.compute_shape(tensor, index)
        env = CompileEnvironment.current()
        return env.new_index_result(tensor, target_shape)
    if isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        return tensor_like.new_empty(target_shape)
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")

@_decorators.register_to_device_ir(load)
def _(
    tracer: "proxy_tensor.PythonKeyTracer",
    tensor: torch.Tensor | tuple[object, ...],
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    """Handle load during tracing, applying prologue if one is registered.

    This handler enables prologue fusion by intercepting load operations
    during tracing and applying prologue transformations (like relu) to
    the loaded value.
    """
    import os
    import sys
    from .._compiler.fusion.convert import get_epilogue_tracing_state, EpilogueOp

    state = get_epilogue_tracing_state()

    # Get prologue info for this tensor if any
    prologue_info = None
    if state is not None and isinstance(tensor, torch.Tensor):
        prologue_info = state.get_prologue_info_for_tensor(id(tensor))
    from ._decorators import args_to_proxies

    # Create the proxy for the load operation
    proxy_out = tracer.create_proxy(
        "call_function",
        load,  # The original load wrapper
        *args_to_proxies(tracer, (tensor, index, extra_mask, eviction_policy), {}),
    )

    # Compute the fake result for shape inference
    if isinstance(tensor, torch.Tensor):
        target_shape = SubscriptIndexing.compute_shape(tensor, index)
        env = CompileEnvironment.current()
        fake_out = env.new_index_result(tensor, target_shape)
    elif isinstance(tensor, tuple):
        tensor_like, dev_ptrs = tensor
        assert isinstance(tensor_like, torch.Tensor)
        assert isinstance(dev_ptrs, torch.Tensor)
        tensor_shape = SubscriptIndexing.compute_shape(tensor_like, index)
        target_shape = list(dev_ptrs.size()) + tensor_shape
        fake_out = tensor_like.new_empty(target_shape)
    else:
        raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")

    proxy_out.node.meta["val"] = fake_out
    proxy_tensor.track_tensor_tree(fake_out, proxy_out, constant=None, tracer=tracer)

    # Apply prologue transformations if any
    if prologue_info is not None:
        result_fake = fake_out
        result_proxy = proxy_out
        for op in prologue_info.ops:
            if op.is_binary:
                # Binary op with external buffer or constant
                if op.constant_val is not None:
                    # Get Scalar overload for the op if available
                    scalar_op_fn = op.op_fn
                    if hasattr(op.op_fn, "_overloadpacket"):
                        try:
                            scalar_op_fn = getattr(op.op_fn._overloadpacket, "Scalar", op.op_fn)
                        except Exception:
                            pass

                    op_result_fake = scalar_op_fn(result_fake, op.constant_val)
                    op_proxy = tracer.create_proxy(
                        "call_function",
                        scalar_op_fn,
                        (result_proxy, op.constant_val),
                        {},
                    )
                elif op.ext_buf is not None:
                    # Binary op with external buffer (e.g., x + bias)
                    from . import _tracing_ops
                    from .._compiler.host_function import HostFunction
                    from .._compiler.variable_origin import ArgumentOrigin
                    from .._compiler.device_ir import APIFuncLowering

                    ext_buf = op.ext_buf

                    # Use dim_idx from the op (computed by extract_prologue_ops from FX graph analysis)
                    # This correctly determines which dimension the 1D buffer broadcasts along
                    dim_idx = op.dim_idx if op.dim_idx >= 0 else 0

                    # Get the tile_idx from the subscript (index)
                    tile_idx = index[dim_idx] if dim_idx < len(index) else index[-1]

                    # Register ext_buf in tensor_to_origin if needed
                    # Use same naming convention as epilogue closures: _ext_{buf_name}
                    host_fn = HostFunction.current()
                    origin_name = f"_ext_{op.ext_buf_name}"
                    if ext_buf not in host_fn.tensor_to_origin:
                        origin = ArgumentOrigin(origin_name)
                        host_fn.tensor_to_origin[ext_buf] = origin

                    # Create a proxy for the external buffer using _host_tensor
                    ext_buf_proxy = tracer.create_proxy(
                        "call_function",
                        _tracing_ops._host_tensor,
                        (origin_name,),
                        {},
                        name=f"ext_{origin_name}",
                    )
                    ext_buf_proxy.node.meta["val"] = ext_buf
                    ext_buf_proxy.node.meta["lowering"] = APIFuncLowering(_tracing_ops._host_tensor)

                    # Convert tile_idx to proxy
                    tile_idx_proxy = args_to_proxies(tracer, (tile_idx,), {})[0][0]

                    # Build subscript based on ext_buf shape and dim_idx
                    ext_buf_ndim = len(ext_buf.shape)

                    # For row broadcast (dim_idx=0), we need shape [u0, 1] not [u0]
                    # For col broadcast (dim_idx=1), we need shape [u1] which broadcasts to [1, u1]
                    needs_unsqueeze = (ext_buf_ndim == 1 and dim_idx == 0 and len(index) > 1)

                    # For 1D buffers, just use tile_idx directly
                    if ext_buf_ndim == 1:
                        ext_subscript = [tile_idx]
                        ext_subscript_proxy = [tile_idx_proxy]
                    else:
                        # For 2D+ buffers, build subscript based on dim_idx
                        ext_subscript = list(index[:ext_buf_ndim])
                        ext_subscript_proxy = list(args_to_proxies(tracer, tuple(ext_subscript), {})[0])

                    # Compute shape for the loaded external buffer
                    ext_target_shape = SubscriptIndexing.compute_shape(ext_buf, ext_subscript)
                    env = CompileEnvironment.current()
                    ext_loaded_fake = env.new_index_result(ext_buf, ext_target_shape)

                    # Create load proxy for external buffer
                    ext_load_proxy = tracer.create_proxy(
                        "call_function",
                        load,
                        (ext_buf_proxy, ext_subscript_proxy, None, None),
                        {},
                    )
                    ext_load_proxy.node.meta["val"] = ext_loaded_fake
                    proxy_tensor.track_tensor_tree(ext_loaded_fake, ext_load_proxy, constant=None, tracer=tracer)

                    # For row broadcast with 1D buffer, unsqueeze to add trailing dimension
                    # This makes [u0] -> [u0, 1] for correct broadcasting with [u0, u1]
                    if needs_unsqueeze:
                        unsqueeze_fake = ext_loaded_fake.unsqueeze(-1)
                        unsqueeze_proxy = tracer.create_proxy(
                            "call_function",
                            torch.ops.aten.unsqueeze.default,
                            (ext_load_proxy, -1),
                            {},
                        )
                        unsqueeze_proxy.node.meta["val"] = unsqueeze_fake
                        proxy_tensor.track_tensor_tree(unsqueeze_fake, unsqueeze_proxy, constant=None, tracer=tracer)
                        ext_loaded_fake = unsqueeze_fake
                        ext_load_proxy = unsqueeze_proxy

                    # Apply the binary operation (e.g., add)
                    op_result_fake = op.op_fn(result_fake, ext_loaded_fake)
                    op_proxy = tracer.create_proxy(
                        "call_function",
                        op.op_fn,
                        (result_proxy, ext_load_proxy),
                        {},
                    )
                else:
                    raise ValueError("Binary op requires constant_val or ext_buf")
            else:
                # Unary op (like relu)
                if op.dtype_val is not None:
                    # to_dtype operation
                    op_result_fake = result_fake.to(op.dtype_val)
                    op_proxy = tracer.create_proxy(
                        "call_function",
                        torch._prims.convert_element_type,
                        (result_proxy, op.dtype_val),
                        {},
                    )
                else:
                    op_result_fake = op.op_fn(result_fake)
                    op_proxy = tracer.create_proxy(
                        "call_function",
                        op.op_fn,
                        (result_proxy,),
                        {},
                    )

            op_proxy.node.meta["val"] = op_result_fake
            proxy_tensor.track_tensor_tree(op_result_fake, op_proxy, constant=None, tracer=tracer)

            result_fake = op_result_fake
            result_proxy = op_proxy

        return result_fake

    return fake_out

@_decorators.codegen(load, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor, subscript, extra_mask = state.proxy_arg(0), state.proxy_arg(1), state.ast_args[2]
    assert isinstance(subscript, (list, tuple)) and isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None
    device_fn, load_idx = state.device_function, state.device_function.device_load_index
    device_fn.device_load_index += 1

    if eviction_policy is None and state.codegen.on_device:
        policies = state.config.load_eviction_policies
        if load_idx < len(policies): eviction_policy = _EVICTION_POLICY_MAP.get(policies[load_idx], policies[load_idx])
    if eviction_policy is not None: eviction_policy = ast.Constant(value=eviction_policy)

    if isinstance(tensor, torch.Tensor):
        from ..language import tile_index
        tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
        if isinstance(tensor_node, torch.fx.Node) and tensor_node.op == "call_function" and tensor_node.target == tile_index:
            env, block_id = CompileEnvironment.current(), CompileEnvironment.current().get_block_id(tensor.size(0))
            assert block_id is not None
            parts = ["None" if idx is None else ":" if idx == slice(None) else (_ for _ in ()).throw(AssertionError(f"Unexpected index type: {idx}")) for idx in subscript]
            return expr_from_string(f"{state.codegen.index_var(block_id)}[{', '.join(parts)}]")

        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        return device_fn.get_indexing_strategy(indexing_idx).codegen_load(state, tensor, [*subscript], extra_mask, eviction_policy)
    if isinstance(tensor, tuple):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state, tensor, dev_ptrs_ast, [*subscript], extra_mask, eviction_policy
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")

@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0

# TODO(joydddd): Add support for stack tensor in ref mode.
@_decorators.ref(load)
def _(
    tensor: torch.Tensor,
    index: list[object],
    extra_mask: torch.Tensor | None = None,
    eviction_policy: str | None = None,
) -> torch.Tensor:
    from .ref_tile import RefTile

    if extra_mask is None:
        # Convert RefTiles to indices
        indices = [idx.index if isinstance(idx, RefTile) else idx for idx in index]
        # Use meshgrid for Cartesian product when we have multiple tensor indices
        tensor_idxs = [
            i for i, idx in enumerate(indices) if isinstance(idx, torch.Tensor)
        ]
        if len(tensor_idxs) > 1:
            # pyrefly: ignore [bad-argument-type]
            grids = torch.meshgrid(*(indices[i] for i in tensor_idxs), indexing="ij")
            for i, grid in zip(tensor_idxs, grids, strict=False):
                indices[i] = grid
        # pyrefly: ignore [bad-argument-type]
        return tensor[tuple(indices)]

    # Create zero result matching mask shape
    result = torch.zeros(extra_mask.shape, dtype=tensor.dtype, device=tensor.device)

    # Process indices: convert RefTiles and clamp tensor indices
    orig_indices, safe_indices, is_tensor_mask = [], [], []
    for i, idx in enumerate(index):
        if isinstance(idx, RefTile):
            idx = idx.index  # Convert RefTile to tensor

        if isinstance(idx, torch.Tensor):
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            orig_indices.append(idx)
            safe_indices.append(torch.clamp(idx, 0, dim_size - 1))
            is_tensor_mask.append(True)
        else:
            orig_indices.append(idx)
            safe_indices.append(idx)
            is_tensor_mask.append(False)

    # Apply broadcasting if we have multiple tensor indices
    tensor_positions = [i for i, is_tensor in enumerate(is_tensor_mask) if is_tensor]

    if len(tensor_positions) > 1:
        # Add unsqueeze operations for broadcasting
        broadcast_indices = []
        for i, (idx, is_tensor) in enumerate(
            zip(safe_indices, is_tensor_mask, strict=False)
        ):
            if is_tensor:
                new_idx = idx
                # Add dimension for each other tensor index
                for j, other_pos in enumerate(tensor_positions):
                    if other_pos != i:
                        new_idx = new_idx.unsqueeze(j if other_pos < i else -1)
                broadcast_indices.append(new_idx)
            else:
                broadcast_indices.append(idx)
        values = tensor[tuple(broadcast_indices)]
    else:
        values = tensor[tuple(safe_indices)]

    # Build validity mask
    valid_mask = extra_mask.clone()
    for i, (orig_idx, is_tensor) in enumerate(
        zip(orig_indices, is_tensor_mask, strict=False)
    ):
        if is_tensor:
            dim_size = tensor.shape[i] if i < len(tensor.shape) else tensor.numel()
            in_bounds = (orig_idx >= 0) & (orig_idx < dim_size)
            # Broadcast to match mask shape by adding dimensions
            # Count how many tensor indices come before and after this one
            n_before = sum(1 for j in range(i) if is_tensor_mask[j])
            n_after = sum(
                1 for j in range(i + 1, len(is_tensor_mask)) if is_tensor_mask[j]
            )

            # Add dimensions: n_after dimensions at the end, n_before at the beginning
            for _ in range(n_after):
                in_bounds = in_bounds.unsqueeze(-1)
            for _ in range(n_before):
                in_bounds = in_bounds.unsqueeze(0)
            valid_mask = valid_mask & in_bounds

    return torch.where(valid_mask, values, result)
