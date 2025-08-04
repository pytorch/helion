from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

import sympy
import torch
from torch._inductor.utils import triton_type

from .. import exc
from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .host_function import HostFunction
from .tile_strategy import DeviceLoopState
from .utils import compute_slice_size
from .utils import get_slice_start
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .device_function import TensorDescriptorArg
    from .inductor_lowering import CodegenState

    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


@dataclasses.dataclass
class NormalizedIndex:
    """Represents a normalized index with all transformations applied."""
    index_type: str  # "int", "slice", "tensor", "none"
    value: int | slice | torch.Tensor
    original: object  # Original index for debugging
    dim_size: int | torch.SymInt  # Size of the dimension being indexed

    @staticmethod
    def normalize_negative_runtime(
        k: int,
        dim_idx: int,
        fake_value: torch.Tensor,
        state: CodegenState,
    ) -> str:
        """Normalize negative indices to positive ones at runtime.

        Args:
            k: The negative index value
            dim_idx: The dimension index
            fake_value: The fake tensor to get dimension size from
            state: The codegen state

        Returns:
            String representation of the normalized index
        """
        assert k < 0, "This function should only be called for negative indices"

        dim_size = fake_value.size(dim_idx)
        # Handle both concrete and symbolic dimension sizes
        if isinstance(dim_size, int):
            normalized_k = k + dim_size
            return repr(normalized_k)
        # For symbolic dimensions, we need to generate the proper expression
        # The state.codegen is a GenerateAST instance which has device_function
        sympy_expr = dim_size._sympy_() + k
        return f"({state.codegen.device_function.user_sympy_expr(sympy_expr)})"


@dataclasses.dataclass
class NormalizedSubscript:
    """Complete normalized subscript for a tensor."""
    indices: list[NormalizedIndex]
    output_shape: list[int | torch.SymInt]
    consumed_dims: int  # How many input dims were consumed


class IndexNormalizer:
    """Pre-processes indices before they reach indexing strategies."""

    def normalize_indices(
        self,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        state: CodegenState,
    ) -> NormalizedSubscript:
        """Main entry point - normalizes all indices in one pass.

        Args:
            fake_tensor: The tensor being indexed
            subscript: List of indices (int, slice, tensor, None, SymInt)
            state: The codegen state

        Returns:
            NormalizedSubscript with all indices normalized
        """
        normalized_indices = []
        input_dims = list(fake_tensor.shape)
        output_shape = []
        consumed_dims = 0
        env = CompileEnvironment.current()

        for k in subscript:
            if consumed_dims < len(input_dims):
                dim_size = input_dims[consumed_dims]
            else:
                # This handles the case where we have more indices than dimensions
                raise exc.InvalidIndexingType(
                    f"Too many indices for tensor of dimension {fake_tensor.ndim}"
                )

            if k is None:
                # None adds a new dimension of size 1
                normalized = NormalizedIndex(
                    index_type="none",
                    value=None,
                    original=k,
                    dim_size=1,
                )
                normalized_indices.append(normalized)
                output_shape.append(1)
                # None doesn't consume a dimension

            elif isinstance(k, int):
                # Normalize negative indices
                normalized_value = self._normalize_int(k, dim_size, state)
                normalized = NormalizedIndex(
                    index_type="int",
                    value=normalized_value,
                    original=k,
                    dim_size=dim_size,
                )
                normalized_indices.append(normalized)
                consumed_dims += 1
                # Integer index removes this dimension from output

            elif isinstance(k, torch.SymInt):
                # Handle symbolic integers
                normalized = NormalizedIndex(
                    index_type="symint",
                    value=k,
                    original=k,
                    dim_size=dim_size,
                )
                normalized_indices.append(normalized)
                # Check if this is a BlockSizeOrigin
                symbol = k._sympy_()
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                    if origin and isinstance(origin.origin, BlockSizeOrigin):
                        if dim_size != 1:
                            output_shape.append(k)
                        else:
                            output_shape.append(1)
                consumed_dims += 1

            elif isinstance(k, slice):
                # Normalize slice bounds
                normalized_slice = self._normalize_slice(k, dim_size, state)
                normalized = NormalizedIndex(
                    index_type="slice",
                    value=normalized_slice,
                    original=k,
                    dim_size=dim_size,
                )
                normalized_indices.append(normalized)
                consumed_dims += 1
                # Calculate output size for this slice
                slice_size = compute_slice_size(normalized_slice, dim_size)
                if slice_size != 1:
                    rdim = env.allocate_reduction_dimension(slice_size)
                    output_shape.append(rdim.var)
                else:
                    output_shape.append(1)

            elif isinstance(k, torch.Tensor):
                # Tensor indices
                normalized = NormalizedIndex(
                    index_type="tensor",
                    value=k,
                    original=k,
                    dim_size=dim_size,
                )
                normalized_indices.append(normalized)
                consumed_dims += 1
                # Tensor indexing adds dimensions from the tensor shape
                if k.ndim == 1 or (len(subscript) == 1 and fake_tensor.ndim == 1):
                    output_shape.extend(k.size())
                else:
                    raise exc.InvalidIndexingType(
                        f"Advanced indexing with tensor {k.shape} not supported"
                    )
            else:
                raise exc.InvalidIndexingType(f"Invalid index type: {type(k)}")


        return NormalizedSubscript(
            indices=normalized_indices,
            output_shape=output_shape,
            consumed_dims=consumed_dims,
        )

    def _normalize_int(self, k: int, dim_size: int | torch.SymInt, state: CodegenState) -> int:
        """Normalize negative integer indices."""
        if k < 0:
            if isinstance(dim_size, int):
                return k + dim_size
            # For symbolic dimensions, we can't normalize at compile time
            # This will be handled by the indexing strategies
            return k
        return k

    def _normalize_slice(self, k: slice, dim_size: int | torch.SymInt, state: CodegenState) -> slice:
        """Normalize slice bounds, handling None and negative values."""
        start = k.start if k.start is not None else 0
        stop = k.stop if k.stop is not None else dim_size
        step = k.step if k.step is not None else 1

        # Normalize negative bounds
        if isinstance(dim_size, int):
            if isinstance(start, int) and start < 0:
                start = start + dim_size
            if isinstance(stop, int) and stop < 0:
                stop = stop + dim_size

        return slice(start, stop, step)




class IndexingStrategy:
    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        raise NotImplementedError

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        raise NotImplementedError

    @staticmethod
    def select(config: Config) -> IndexingStrategy:
        indexing = config.indexing
        if indexing == "pointer":
            return PointerIndexingStrategy()
        if indexing == "tensor_descriptor":
            return TensorDescriptorIndexingStrategy()
        if indexing == "block_ptr":
            return BlockPtrIndexingStrategy()
        raise RuntimeError(
            f"Invalid indexing strategy: {indexing!r}, "
            "must be one of 'pointer', 'tensor_descriptor', 'block_ptr'"
        )


class PointerIndexingStrategy(IndexingStrategy):
    """Generate the original pointer math to load/store from tensors"""

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        extra = ""
        if indexing.has_mask():
            # For FP8 dtypes, use other=0.0 (float literal) instead of other=0 (int literal)
            # because Triton cannot cast integer 0 to FP8 types
            if fake_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                extra = ", other=0.0"
            else:
                extra = ", other=0"
        name = state.device_function.tensor_arg(fake_tensor).name
        return expr_from_string(
            f"tl.load({name} + offset, mask{extra})",
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        name = state.device_function.tensor_arg(fake_tensor).name

        # Check if value is a tensor load (Name node with id matching a tensor arg)
        if isinstance(value, ast.Name) and hasattr(
            state.device_function, "_tensor_args"
        ):
            # Check if this name corresponds to a tensor argument
            tensor = None
            for t, tensor_arg in state.device_function._tensor_args.items():
                if tensor_arg.name == value.id:
                    tensor = t
                    break

            if tensor is not None:
                # Get the shape of the slice we're storing to
                output_shape = SubscriptIndexing.compute_shape(fake_tensor, subscript)
                if len(output_shape) == 1 and tensor.ndim == 1:
                    # Load the entire 1D tensor
                    value_indexing = SubscriptIndexing.create(
                        state, tensor, [slice(None)], None
                    )
                    value = expr_from_string(
                        f"tl.load({value.id} + offset, mask)",
                        offset=value_indexing.index_expr,
                        mask=value_indexing.mask_expr,
                    )

        return expr_from_string(
            f"tl.store({name} + offset, value, mask)",
            value=value,
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )


def _has_non_power_of_2_slice(fake_tensor: torch.Tensor, subscript: list[Any]) -> bool:
    """Check if subscript contains non-power-of-2 slices that aren't full slices."""
    from .utils import compute_slice_size
    from .utils import get_slice_start
    
    for k in subscript:
        if isinstance(k, slice):
            size = fake_tensor.size(subscript.index(k)) if subscript.index(k) < fake_tensor.ndim else 1
            start = get_slice_start(k)
            slice_size = compute_slice_size(k, size)
            is_full_slice = (start == 0 and slice_size == size)
            if not is_full_slice and slice_size != 1:
                if slice_size & (slice_size - 1) != 0:
                    return True
    return False


class BlockPtrIndexingStrategy(IndexingStrategy):
    """Use block_ptr to load/store from tensors"""

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        if _has_non_power_of_2_slice(fake_tensor, subscript) or not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return PointerIndexingStrategy().codegen_load(
                state, fake_tensor, subscript, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)
        return indexing.reshape_load(
            state,
            expr_from_string(
                f"tl.load(block_ptr, boundary_check={indexing.boundary_check(state)}, padding_option='zero')",
                block_ptr=indexing.make_block_ptr(state),
            ),
        )

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        if _has_non_power_of_2_slice(fake_tensor, subscript) or not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return PointerIndexingStrategy().codegen_store(
                state, fake_tensor, subscript, value, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)

        # Check if value is a tensor argument that needs to be loaded
        if isinstance(value, ast.Name):
            # Check if this is a tensor argument
            df = state.device_function
            tensor_arg = None
            for arg in df.arguments:
                if hasattr(arg, "name") and arg.name == value.id:
                    from .device_function import TensorArg
                    if isinstance(arg, TensorArg):
                        tensor_arg = arg
                        break

            if tensor_arg:
                # This is a tensor argument - we need to load it with block_ptr
                val_tensor = tensor_arg.fake_value
                val_indexing = BlockedSubscriptIndexing(
                    val_tensor,
                    reshaped_size=list(val_tensor.shape),
                )
                # Add offsets and block_shape for all dimensions
                env = CompileEnvironment.current()
                for dim in range(val_tensor.ndim):
                    size = val_tensor.size(dim)
                    if size != 1:
                        rdim = env.allocate_reduction_dimension(size)
                        val_indexing.offsets.append("0")
                        val_indexing.block_shape.append(rdim.var)
                    else:
                        val_indexing.offsets.append("0")
                        val_indexing.block_shape.append(1)

                # Load the tensor value
                value = expr_from_string(
                    f"tl.load(val_block_ptr, boundary_check={val_indexing.boundary_check(state)}, padding_option='zero')",
                    val_block_ptr=val_indexing.make_block_ptr(state),
                )

        return expr_from_string(
            f"tl.store(block_ptr, value, boundary_check={indexing.boundary_check(state)})",
            block_ptr=indexing.make_block_ptr(state),
            value=indexing.reshape_store(state, value),
        )


class TensorDescriptorIndexingStrategy(IndexingStrategy):
    """Use TensorDescriptor to load/store from tensors"""

    @staticmethod
    def is_supported(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> bool:
        """Check if tensor descriptor indexing is supported with additional requirements."""
        # First check the basic BlockedSubscriptIndexing requirements
        if not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return False

        # Additional tensor descriptor requirements:
        # 1) ndim must be between 2 and 5
        if not (2 <= fake_tensor.ndim <= 5):
            return False

        # 2) Exactly 1 dimension should have stride==1
        env = CompileEnvironment.current()
        stride_one_count = 0
        element_size = fake_tensor.element_size()
        for dim in range(fake_tensor.ndim):
            stride = env.size_hint(fake_tensor.stride(dim))
            if stride == 1:
                stride_one_count += 1
            else:
                # 3) All other dimensions should have 16-byte aligned strides
                byte_stride = stride * element_size
                if byte_stride % 16 != 0:
                    return False
        if stride_one_count != 1:
            # There should be exactly one dimension with stride==1
            return False

        def valid_block_size(
            block_size: int | torch.SymInt | None, stride: int | torch.SymInt, idx: int
        ) -> bool:
            if not isinstance(block_size, int):
                return False

            if (
                get_tensor_descriptor_fn_name()
                == "tl._experimental_make_tensor_descriptor"
            ):
                # https://github.com/triton-lang/triton/blob/d654e0f2d91f07496454e0fcbec2a9b97df37d47/python/triton/language/semantic.py#L1162
                threshold = 32 // fake_tensor.dtype.itemsize
                if idx == 0:
                    threshold = min(8, threshold)

                if fake_tensor.ndim == 2 and block_size < threshold:
                    return False

            # was getting some IMAs with small block sizes even in non-stride 1 dims
            return block_size * element_size >= 16 or (block_size == 1 and stride != 1)

        # 4) Check minimum 16 bytes in each dimension
        sizes = fake_tensor.size()
        strides = fake_tensor.stride()
        size_stride = collections.deque(zip(sizes, strides, strict=True))
        config = DeviceFunction.current().config
        for i, k in enumerate(subscript):
            if k is None:
                continue
            size, stride = size_stride.popleft()
            if isinstance(k, slice):
                # Slices with steps are not supported in tensor descriptor mode
                if k.step is not None and k.step != 1:
                    return False
                block_size = env.allocate_reduction_dimension(size).from_config(config)
                if not valid_block_size(block_size, stride, i):
                    return False
            elif isinstance(k, torch.SymInt):
                block_id = env.get_block_id(k)
                if block_id is None:
                    return False
                block_size = env.block_sizes[block_id].from_config(config)
                if not valid_block_size(block_size, stride, i):
                    return False

        return True

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        if not self.is_supported(state, fake_tensor, subscript, extra_mask):
            return PointerIndexingStrategy().codegen_load(
                state, fake_tensor, subscript, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)

        # Load from tensor descriptor with permuted offsets
        load_expr = expr_from_string(
            f"{indexing.tensor_descriptor(state)}.load({indexing.offsets_str_permuted(state)})"
        )

        # Apply inverse permutation to the loaded result if needed
        desc_arg = indexing.tensor_descriptor_arg(state)
        if desc_arg.permutation is not None:
            load_expr = expr_from_string(
                f"tl.permute(load_result, {desc_arg.inverse_permutation!r})",
                load_result=load_expr,
            )

        return indexing.reshape_load(state, load_expr)

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        if not self.is_supported(state, fake_tensor, subscript, extra_mask):
            return PointerIndexingStrategy().codegen_store(
                state, fake_tensor, subscript, value, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)

        # Apply permutation to the value being stored if needed
        desc_arg = indexing.tensor_descriptor_arg(state)
        store_value = indexing.reshape_store(state, value)

        if desc_arg.permutation is not None:
            # Apply permutation to the value
            store_value = expr_from_string(
                f"tl.permute(store_val, {desc_arg.permutation!r})",
                store_val=store_value,
            )

        return expr_from_string(
            f"{indexing.tensor_descriptor(state)}.store({indexing.offsets_str_permuted(state)}, value)",
            value=store_value,
        )


class StackIndexingStrategy:
    """
    Generate pointer math for stacking load/store to several device memory pointers sharing the same indexing.

    offset, mask are calculated for the tensor_like template tensor and then broadcasted to each dev_ptr
    , with the results stacked.

    e.g. for a 1D offset tensor and a 1D dev_ptr array, the stack offset is:
    stack_offset = dev_ptrs[:, None] + offset[None, :]

    """

    @staticmethod
    def get_broadcast_str(
        stack_shape: ShapeLike,
        subscript_shape: ShapeLike,
    ) -> tuple[str, str]:
        """
        Args:
            stack_shape: shape of the dev_ptr tensor.
            subscript_shape: shape of subscription for each individual tensor.

        Returns:
            the broadcast str for dev_ptrs and individual tensor offset.
        """
        stack_broadcast_keys = [":" for _ in stack_shape] + [
            "None" for _ in subscript_shape
        ]
        stack_broadcast = f"[{', '.join(stack_broadcast_keys)}]"
        tensor_broadcast_keys = ["None" for _ in stack_shape] + [
            ":" for _ in subscript_shape
        ]
        tensor_broadcast = f"[{', '.join(tensor_broadcast_keys)}]"

        return stack_broadcast, tensor_broadcast

    @staticmethod
    def get_mask_expr(
        state: CodegenState,
        indexing: SubscriptIndexing,
        stack_shape: ShapeLike,
        subscript_shape: ShapeLike,
    ) -> ast.AST | None:
        stack_broadcast, tensor_broadcast = StackIndexingStrategy.get_broadcast_str(
            stack_shape, subscript_shape
        )

        mask_exprs = []
        dev_ptr_mask_exprs = []
        # Generate Mask

        for dim, size in enumerate(stack_shape):
            if (
                index := CompileEnvironment.current().get_block_id(size)
            ) is not None and (mask_var := state.codegen.mask_var(index)) is not None:
                expand = state.tile_strategy.expand_str(stack_shape, dim)
                dev_ptr_mask_exprs.append(f"({mask_var}{expand})")

        if dev_ptr_mask_exprs:
            dev_ptr_mask_expr = f"({'&'.join(dev_ptr_mask_exprs)})"
            if len(dev_ptr_mask_exprs) < len(stack_shape):
                dev_ptr_mask_expr = f"tl.broadcast_to({dev_ptr_mask_expr}, {state.tile_strategy.shape_str(stack_shape)})"
            dev_ptr_mask_expr = f"({dev_ptr_mask_expr}){stack_broadcast}"
            mask_exprs.append(dev_ptr_mask_expr)

        if indexing.has_mask():
            mask_exprs.append(f"(tensor_mask){tensor_broadcast}")
            return expr_from_string(
                "&".join(mask_exprs), tensor_mask=indexing.mask_expr
            )
        if mask_exprs:
            return expr_from_string("&".join(mask_exprs))
        return None

    @staticmethod
    def codegen_load(
        state: CodegenState,
        stack_tensor: tuple[torch.Tensor, torch.Tensor],
        dev_ptrs_ast: ast.AST,
        subscript: list[object],
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        tensor_like, dev_ptrs = stack_tensor
        indexing = SubscriptIndexing.create(state, tensor_like, subscript, extra_mask)
        subscripts_shape = SubscriptIndexing.compute_shape(tensor_like, subscript)
        stack_shape = [*dev_ptrs.size()]

        mask_expr = StackIndexingStrategy.get_mask_expr(
            state, indexing, stack_shape, subscripts_shape
        )
        extra = ", other=0"
        if mask_expr is None:
            mask_expr = expr_from_string("None")
            extra = ""

        stack_broadcast, tensor_broadcast = StackIndexingStrategy.get_broadcast_str(
            stack_shape, subscripts_shape
        )

        dtype = triton_type(tensor_like.dtype)
        return expr_from_string(
            f"tl.load((base.to(tl.pointer_type({dtype}))){stack_broadcast} + (offset){tensor_broadcast}, mask{extra})",
            base=dev_ptrs_ast,
            offset=indexing.index_expr,
            mask=mask_expr,
        )

    @staticmethod
    def codegen_store(
        state: CodegenState,
        stack_tensor: tuple[torch.Tensor, torch.Tensor],
        dev_ptrs_ast: ast.AST,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        tensor_like, dev_ptrs = stack_tensor
        indexing = SubscriptIndexing.create(state, tensor_like, subscript, extra_mask)
        subscripts_shape = SubscriptIndexing.compute_shape(tensor_like, subscript)
        stack_shape = [*dev_ptrs.size()]

        mask_expr = StackIndexingStrategy.get_mask_expr(
            state, indexing, stack_shape, subscripts_shape
        )
        if mask_expr is None:
            mask_expr = expr_from_string("None")

        stack_broadcast, tensor_broadcast = StackIndexingStrategy.get_broadcast_str(
            stack_shape, subscripts_shape
        )

        dtype = triton_type(tensor_like.dtype)
        return expr_from_string(
            f"tl.store(base.to(tl.pointer_type({dtype})){stack_broadcast} + (offset){tensor_broadcast}, value, mask)",
            base=dev_ptrs_ast,
            value=value,
            offset=indexing.index_expr,
            mask=mask_expr,
        )


class SubscriptIndexing(NamedTuple):
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )

    @staticmethod
    def _generate_slice_index(
        start: int | torch.SymInt,
        index_var: str,
        expand: str,
        step: int | None = None,
    ) -> str:
        """Generate slice index expression with optional step."""
        if step is not None:
            # Strided index: start + index * step
            return f"({start} + ({index_var}) * {step}){expand}"
        if start != 0:
            # Index with offset: start + index
            return f"({start} + ({index_var})){expand}"
        # Simple index
        return f"({index_var}){expand}"

    @staticmethod
    def compute_shape(
        tensor: torch.Tensor, index: list[object]
    ) -> list[int | torch.SymInt]:
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(index, (list, tuple)), index
        input_size: collections.deque[int | torch.SymInt] = collections.deque(
            tensor.size()
        )
        output_size = []
        env = CompileEnvironment.current()
        for k in index:
            if k is None:
                output_size.append(1)
            elif isinstance(k, int):
                input_size.popleft()
            elif isinstance(k, torch.SymInt):
                input_size.popleft()
                symbol = k._sympy_()
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                    if origin and isinstance(origin.origin, BlockSizeOrigin):
                        if tensor.size(tensor.ndim - len(input_size) - 1) != 1:
                            output_size.append(k)
                        else:
                            output_size.append(1)
            elif isinstance(k, slice):
                size = input_size.popleft()
                # Handle slices with steps
                slice_size = compute_slice_size(k, size)
                if slice_size != 1:
                    rdim = env.allocate_reduction_dimension(slice_size)
                    output_size.append(rdim.var)
                else:
                    output_size.append(1)
            elif isinstance(k, torch.Tensor) and (
                k.ndim == 1 or (len(index) == 1 and tensor.ndim == 1)
            ):
                input_size.popleft()
                output_size.extend(k.size())
            else:
                raise exc.InvalidIndexingType(k)
        assert len(input_size) == 0, "invalid subscript"
        return output_size

    @staticmethod
    def create(
        state: CodegenState,
        fake_value: torch.Tensor,
        index: list[object],
        extra_mask: ast.AST | None = None,
    ) -> SubscriptIndexing:
        # Use IndexNormalizer to pre-process indices
        normalizer = IndexNormalizer()
        normalized = normalizer.normalize_indices(fake_value, index, state)

        tile_strategy = state.tile_strategy
        output_idx = 0
        index_values = []
        mask_values = {}
        output_size = normalized.output_shape
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()

        # Track position in the original tensor dimensions
        tensor_dim_idx = 0

        for norm_idx in normalized.indices:
            if norm_idx.index_type == "none":
                # None adds a dimension, doesn't consume tensor dimension
                output_idx += 1
            elif norm_idx.index_type == "int":
                # Integer index - use normalized value
                if norm_idx.value < 0:
                    # Handle negative indices that couldn't be normalized at compile time
                    index_values.append(NormalizedIndex.normalize_negative_runtime(
                        norm_idx.value, tensor_dim_idx, fake_value, state
                    ))
                else:
                    index_values.append(repr(norm_idx.value))
                tensor_dim_idx += 1
            elif norm_idx.index_type == "symint":
                # SymInt index
                k = norm_idx.value
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)

                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    index_var = state.codegen.index_var(origin.origin.block_id)
                    expand = tile_strategy.expand_str(output_size, output_idx)
                    index_values.append(f"({index_var}){expand}")
                    if (
                        mask := state.codegen.mask_var(origin.origin.block_id)
                    ) and norm_idx.dim_size != 1:
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                else:
                    # Scalar SymInt
                    val = state.device_function.literal_expr(k)
                    index_values.append(f"({val})")
                tensor_dim_idx += 1
            elif norm_idx.index_type == "slice":
                # Slice index
                k = norm_idx.value
                expand = tile_strategy.expand_str(output_size, output_idx)

                # Check if this slice produces output dimensions
                slice_size = compute_slice_size(k, norm_idx.dim_size)
                if slice_size != 1:
                    # Multi-element slice
                    rdim = env.allocate_reduction_dimension(slice_size)
                    block_idx = rdim.block_id
                    index_var = state.codegen.index_var(block_idx)

                    if k.step is not None and k.step != 1:
                        # Strided slice
                        index_values.append(
                            SubscriptIndexing._generate_slice_index(k.start, index_var, expand, k.step)
                        )
                    else:
                        # Regular slice
                        index_values.append(
                            SubscriptIndexing._generate_slice_index(k.start, index_var, expand)
                        )

                    if mask := state.codegen.mask_var(block_idx):
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                else:
                    # Single element slice
                    index_values.append(f"{k.start}{expand}")
                    output_idx += 1
                tensor_dim_idx += 1
            elif norm_idx.index_type == "tensor":
                # Tensor index
                k = norm_idx.value
                expand = tile_strategy.expand_str(output_size, output_idx)
                ast_index = state.ast_args[1]
                assert isinstance(ast_index, (list, tuple))

                if k.ndim == 1:
                    # Find the index of this tensor in the original indices
                    tensor_idx = None
                    for i, orig_idx in enumerate(index):
                        if orig_idx is k:
                            tensor_idx = i
                            break
                    assert tensor_idx is not None
                    index_var = state.codegen.lift(ast_index[tensor_idx], prefix="index").id
                    index_values.append(f"({index_var}){expand}")
                    if (block_idx := env.get_block_id(output_size[output_idx])) is not None:
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                elif len(index) == 1 and fake_value.ndim == 1:
                    # Special case for 1D tensor indexing 1D tensor
                    index_var = state.codegen.lift(ast_index[0], prefix="index").id
                    index_values.append(index_var)
                    output_idx += k.ndim
                    for i, s in enumerate(output_size):
                        if (block_idx := env.get_block_id(s)) is not None and (
                            mask := state.codegen.mask_var(block_idx)
                        ):
                            mask_values.setdefault(
                                f"({mask}){tile_strategy.expand_str(output_size, i)}"
                            )
                tensor_dim_idx += 1
            else:
                raise exc.InvalidIndexingType(f"Unknown normalized index type: {norm_idx.index_type}")


        assert len(output_size) == output_idx
        assert len(index_values) == fake_value.ndim
        index_expr = []
        for i, idx in enumerate(index_values):
            if fake_value.size(i) != 1:
                stride = state.device_function.tensor_stride(fake_value, i).name
                index_expr.append(f"{idx} * {stride}")
        if not index_expr:
            shape_str = tile_strategy.shape_str(output_size)
            index_expr.append(f"tl.zeros({shape_str}, {dtype})")

        kwargs = {}
        if extra_mask is not None:
            mask_values.setdefault("_extra_mask")
            kwargs["_extra_mask"] = extra_mask
        return SubscriptIndexing(
            expr_from_string("+".join(index_expr)),
            expr_from_string("&".join(mask_values) or "None", **kwargs),
        )


@dataclasses.dataclass
class BlockedSubscriptIndexing:
    """Indexing used for block_ptr and tensor_descriptor"""

    base: torch.Tensor

    # properties of the loaded block
    offsets: list[str] = dataclasses.field(default_factory=list)
    block_shape: list[int | torch.SymInt] = dataclasses.field(default_factory=list)
    reshaped_size: list[int | torch.SymInt] = dataclasses.field(default_factory=list)

    def make_block_ptr(self, state: CodegenState) -> ast.AST:
        name = state.device_function.tensor_arg(self.base).name
        fn = state.device_function
        shape = ", ".join(
            [fn.tensor_size(self.base, i).name for i in range(self.base.ndim)]
        )
        strides = ", ".join(
            [fn.tensor_stride(self.base, i).name for i in range(self.base.ndim)]
        )
        block_shape = state.tile_strategy.shape_str(self.block_shape)
        return expr_from_string(
            f"tl.make_block_ptr({name}, [{shape}], [{strides}], {self.offsets_str()}, {block_shape}, {self.order!r})",
        )

    def tensor_descriptor(self, state: CodegenState) -> str:
        return state.device_function.tensor_descriptor_arg(
            self.base, self.block_shape
        ).name

    def tensor_descriptor_arg(self, state: CodegenState) -> TensorDescriptorArg:
        return state.device_function.tensor_descriptor_arg(self.base, self.block_shape)

    def offsets_str(self) -> str:
        return f"[{', '.join(self.offsets)}]"

    def offsets_str_permuted(self, state: CodegenState) -> str:
        """Get offsets string with permutation applied if needed."""
        desc_arg = self.tensor_descriptor_arg(state)
        if desc_arg.permutation is not None:
            # Apply permutation to offsets
            permuted_offsets = [self.offsets[i] for i in desc_arg.permutation]
            return f"[{', '.join(permuted_offsets)}]"
        return self.offsets_str()

    @property
    def ndim(self) -> int:
        return self.base.ndim

    @property
    def order(self) -> list[int]:
        hint = CompileEnvironment.current().size_hint
        stride = sorted([(hint(s), -i, i) for i, s in enumerate(self.base.stride())])
        result = [-1 for _ in stride]
        for order, (_, _, i) in enumerate(stride):
            result[i] = order
        return result

    def boundary_check(self, state: CodegenState) -> str:
        result = []
        for order, size in enumerate(self.block_shape):
            if not (isinstance(size, int) and size == 1):
                # TODO(jansel): we should be able to filter with something like:
                # block_idx = TileStrategy.get_block_index(size)
                # if block_idx is None or state.tile_strategy.need_mask(block_idx):
                result.append(order)
        if result:
            return repr(result)
        return "None"

    def need_reshape(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            # Don't reshape scalar constants - they will be broadcast automatically
            return False
        if len(self.reshaped_size) != len(self.block_shape):
            return True
        env = CompileEnvironment.current()
        for a, b in zip(self.reshaped_size, self.block_shape, strict=True):
            if not env.known_equal(a, b):
                return True
        return False

    def reshape_load(self, state: CodegenState, node: ast.AST) -> ast.AST:
        if not self.need_reshape(node):
            return node
        shape = state.tile_strategy.shape_str(self.reshaped_size)
        return expr_from_string(f"tl.reshape(node, {shape})", node=node)

    def reshape_store(self, state: CodegenState, node: ast.AST) -> ast.AST:
        if not self.need_reshape(node):
            return node
        shape = state.tile_strategy.shape_str(self.block_shape)
        return expr_from_string(f"tl.reshape(node, {shape})", node=node)

    @staticmethod
    def is_supported(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        index: list[object],
        extra_mask: ast.AST | None,
    ) -> bool:
        if extra_mask is not None:
            # TODO(jansel): support block_ptr with extra_mask
            return False
        input_sizes: collections.deque[int | torch.SymInt] = collections.deque(
            fake_tensor.size()
        )
        for k in index:
            if k is None:
                input_size = 1
            else:
                input_size = input_sizes.popleft()
            if isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    block_index = origin.origin.block_id
                    try:
                        state.codegen.offset_var(block_index)
                    except NotImplementedError:
                        return False
                    loop_state = state.codegen.active_device_loops[block_index][-1]
                    if isinstance(loop_state, DeviceLoopState):
                        """
                        Check for a corner case where the loop size does not match the tensor size.
                        In this case, the block masking will be incorrect.  So we check if the
                        masking is needed and bail if it is.
                        """
                        if not loop_state.block_id_to_info[block_index].is_end_matching(
                            input_size
                        ):
                            assert state.fx_node is not None
                            if "masked_value" in state.fx_node.meta:
                                # TODO(jansel): in this case we should be able to lower to block_ptr+tl.where
                                # see test/test_loops.py::TestLoops::test_data_dependent_bounds2
                                return False
            if isinstance(k, torch.Tensor):
                # indirect loads don't work with block_ptr
                return False
            # Slices are supported - we'll decompose non-power-of-2 if needed
        output_shape = SubscriptIndexing.compute_shape(fake_tensor, index)
        return len(output_shape) != 0

    def validate(self) -> None:
        n = self.ndim
        assert len(self.offsets) == n, (
            f"invalid indexing expected {n} dims, got {len(self.offsets)}"
        )
        assert len(self.block_shape) == n, (
            f"invalid indexing expected {n} dims, got {len(self.block_shape)}"
        )

    @staticmethod
    def create(
        state: CodegenState, fake_value: torch.Tensor, index: list[object]
    ) -> BlockedSubscriptIndexing:
        # Use IndexNormalizer to pre-process indices
        normalizer = IndexNormalizer()
        normalized = normalizer.normalize_indices(fake_value, index, state)

        res = BlockedSubscriptIndexing(
            fake_value,
            reshaped_size=normalized.output_shape,
        )
        env = CompileEnvironment.current()

        # Track position in the original tensor dimensions
        tensor_dim_idx = 0

        for norm_idx in normalized.indices:
            if norm_idx.index_type == "none":
                # None is handled by reshaped_size
                pass
            elif norm_idx.index_type == "int":
                # Integer index
                if norm_idx.value < 0:
                    # Handle negative indices that couldn't be normalized at compile time
                    normalized_val = NormalizedIndex.normalize_negative_runtime(
                        norm_idx.value, tensor_dim_idx, fake_value, state
                    )
                    res.offsets.append(normalized_val)
                else:
                    res.offsets.append(repr(norm_idx.value))
                res.block_shape.append(1)
                tensor_dim_idx += 1
            elif norm_idx.index_type == "symint":
                # SymInt index
                k = norm_idx.value
                symbol = k._sympy_()
                origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    if norm_idx.dim_size != 1:
                        res.offsets.append(
                            state.codegen.offset_var(origin.origin.block_id)
                        )
                        res.block_shape.append(k)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
                else:
                    res.offsets.append(state.device_function.literal_expr(k))
                    res.block_shape.append(1)
                tensor_dim_idx += 1
            elif norm_idx.index_type == "slice":
                # Slice index
                k = norm_idx.value
                size = norm_idx.dim_size

                # Handle slices with steps
                if k.step is not None and k.step != 1:
                    # Slices with steps are not supported in block_ptr mode
                    raise exc.InvalidIndexingType(
                        f"Strided slices not supported in block_ptr mode: {k}"
                    )

                # Calculate slice parameters
                start = get_slice_start(k)
                slice_size = compute_slice_size(k, size)

                # Check if this is a full slice ([:] or equivalent)
                is_full_slice = (start == 0 and slice_size == size)

                if is_full_slice:
                    # Full slice - use the entire dimension
                    if size != 1:
                        rdim = env.allocate_reduction_dimension(size)
                        res.offsets.append(state.codegen.offset_var(rdim.block_id))
                        res.block_shape.append(rdim.var)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
                else:
                    # Partial slice
                    if start != 0:
                        res.offsets.append(repr(start))
                    else:
                        res.offsets.append("0")

                    if slice_size != 1:
                        rdim = env.allocate_reduction_dimension(slice_size)
                        res.block_shape.append(rdim.var)
                    else:
                        res.block_shape.append(1)
                tensor_dim_idx += 1
            elif norm_idx.index_type == "tensor":
                # Tensor indices not supported in block_ptr
                raise exc.InvalidIndexingType("Tensor indices not supported in block_ptr mode")
            else:
                raise exc.InvalidIndexingType(f"Unknown normalized index type: {norm_idx.index_type}")


        res.validate()
        return res
