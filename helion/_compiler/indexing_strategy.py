from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch
from torch._inductor.utils import triton_type
from torch._prims_common import compute_required_storage_length
from triton import next_power_of_2

from .. import exc
from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .host_function import HostFunction
from .tile_strategy import DeviceLoopState
from .utils import compute_slice_size
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import IndexingLiteral
    from .device_function import TensorDescriptorArg
    from .inductor_lowering import CodegenState

    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


def _get_padded_iota_original_length(
    state: CodegenState, index_position: int
) -> int | None:
    """Get the original length of a padded iota node at the given index position.

    Args:
        state: The codegen state containing fx_node information
        index_position: The position in the index list to check

    Returns:
        The original (unpadded) length if the index is a padded iota, None otherwise
    """
    try:
        index_node = state.fx_node.args[1][index_position]  # type: ignore[union-attr, index]
        if (
            isinstance(index_node, torch.fx.Node)
            and index_node.target == torch.ops.prims.iota.default  # pyright: ignore[reportAttributeAccessIssue]
            and isinstance(length_arg := index_node.args[0], int)
            and length_arg != next_power_of_2(length_arg)
        ):
            return length_arg
    except (AttributeError, IndexError, TypeError):
        pass

    return None


def _get_tile_with_offset_info(
    k: object, state: CodegenState, k_index: int
) -> tuple[int, int | torch.SymInt] | None:
    """Check if k is a tensor marked as tile.index + offset, return (block_id, offset) if so.

    Args:
        k: The subscript element (fake value)
        state: The codegen state containing the FX node
        k_index: The index of k in the subscript list
    """
    if not isinstance(k, torch.Tensor):
        return None

    # During codegen, we don't have proxy mode, but we have the FX graph
    # The state.fx_node is the load/store node, and its second argument (args[1])
    # is the list of subscript indices as FX nodes
    if state.fx_node is None:
        return None

    # Get the subscript list from the FX node's arguments
    # args[0] is the tensor, args[1] is the subscript list
    if len(state.fx_node.args) < 2:
        return None

    subscript_arg = state.fx_node.args[1]
    if not isinstance(subscript_arg, (list, tuple)):
        return None

    # Find the FX node corresponding to this subscript element
    if k_index >= len(subscript_arg):
        return None

    fx_subscript_node = subscript_arg[k_index]
    if not isinstance(fx_subscript_node, torch.fx.Node):
        return None

    # Check if this FX node has the tile_with_offset metadata
    meta = fx_subscript_node.meta.get("tile_with_offset")
    if meta is not None:
        return (meta["block_id"], meta["offset"])

    return None


class IndexingStrategy:
    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
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
    def select(indexing_literal: IndexingLiteral) -> IndexingStrategy:
        if indexing_literal == "pointer":
            return PointerIndexingStrategy()
        if indexing_literal == "tensor_descriptor":
            return TensorDescriptorIndexingStrategy()
        if indexing_literal == "block_ptr":
            return BlockPtrIndexingStrategy()
        raise RuntimeError(
            f"Invalid indexing strategy: {indexing_literal!r}, "
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
        eviction_policy: ast.AST | None,
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
        extra += ", eviction_policy={ev}" if eviction_policy is not None else ""
        return expr_from_string(
            f"tl.load({name} + {{offset}}, {{mask}}{extra})",
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
            ev=eviction_policy,  # pyright: ignore[reportArgumentType]
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
        return expr_from_string(
            f"tl.store({name} + {{offset}}, {{value}}, {{mask}})",
            value=value,
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )


class BlockPtrIndexingStrategy(IndexingStrategy):
    """Use block_ptr to load/store from tensors"""

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        if not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return PointerIndexingStrategy().codegen_load(
                state, fake_tensor, subscript, extra_mask, eviction_policy
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)
        extra = ", eviction_policy={ev}" if eviction_policy is not None else ""
        return indexing.reshape_load(
            state,
            expr_from_string(
                f"tl.load({{block_ptr}}, boundary_check={indexing.boundary_check(state)}, padding_option='zero'{extra})",
                block_ptr=indexing.make_block_ptr(state),
                ev=eviction_policy,  # pyright: ignore[reportArgumentType]
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
        if not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return PointerIndexingStrategy().codegen_store(
                state, fake_tensor, subscript, value, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)
        return expr_from_string(
            f"tl.store({{block_ptr}}, {{value}}, boundary_check={indexing.boundary_check(state)})",
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

            # Tensor-descriptor path (TMA + WGMMA / stmatrix writes)
            # moves data in 16-byte chunks. Enforce a 16-byte minimum so the
            # generated stores stay aligned and avoid misaligned-address errors.
            return block_size * element_size >= 16

        # 4) Check minimum 16 bytes in each dimension
        sizes = fake_tensor.size()
        strides = fake_tensor.stride()
        size_stride = collections.deque(zip(sizes, strides, strict=True))
        config = DeviceFunction.current().config
        k_index = 0  # Track position for finding FX nodes
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
                k_index += 1
            elif (
                tile_info := _get_tile_with_offset_info(k, state, k_index)
            ) is not None:
                # Tensor marked as tile.index + offset
                block_id, _ = tile_info
                block_size = env.block_sizes[block_id].from_config(config)
                if not valid_block_size(block_size, stride, i):
                    return False
                k_index += 1
            elif isinstance(k, torch.SymInt):
                block_id = env.get_block_id(k)
                if block_id is None:
                    return False
                block_size = env.block_sizes[block_id].from_config(config)
                if not valid_block_size(block_size, stride, i):
                    return False
                k_index += 1

        return True

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        if not self.is_supported(state, fake_tensor, subscript, extra_mask):
            return PointerIndexingStrategy().codegen_load(
                state, fake_tensor, subscript, extra_mask, eviction_policy
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
                f"tl.permute({{load_result}}, {desc_arg.inverse_permutation!r})",
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
                f"tl.permute({{store_val}}, {desc_arg.permutation!r})",
                store_val=store_value,
            )

        return expr_from_string(
            f"{indexing.tensor_descriptor(state)}.store({indexing.offsets_str_permuted(state)}, {{value}})",
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
    def get_element_broadcast_slice(dim_index: int, total_dims: int) -> str:
        broadcast_keys = ["None"] * total_dims
        broadcast_keys[dim_index] = ":"
        return f"[{', '.join(broadcast_keys)}]"

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
            mask_exprs.append(f"({{tensor_mask}}){tensor_broadcast}")
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
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        tensor_like, dev_ptrs = stack_tensor
        indexing = SubscriptIndexing.create(state, tensor_like, subscript, extra_mask)
        subscripts_shape = SubscriptIndexing.compute_shape(
            tensor_like, subscript, state
        )
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
        extra += ", eviction_policy={ev}" if eviction_policy is not None else ""
        return expr_from_string(
            f"tl.load(({{base}}.to(tl.pointer_type({dtype}))){stack_broadcast} + ({{offset}}){tensor_broadcast}, {{mask}}{extra})",
            base=dev_ptrs_ast,
            offset=indexing.index_expr,
            mask=mask_expr,
            ev=eviction_policy,  # pyright: ignore[reportArgumentType]
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
        subscripts_shape = SubscriptIndexing.compute_shape(
            tensor_like, subscript, state
        )
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
            f"tl.store({{base}}.to(tl.pointer_type({dtype})){stack_broadcast} + ({{offset}}){tensor_broadcast}, {{value}}, {{mask}})",
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
    def compute_shape(
        tensor: torch.Tensor, index: list[object], state: CodegenState | None = None
    ) -> list[int | torch.SymInt]:
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(index, (list, tuple)), index
        input_size = collections.deque(tensor.size())
        output_size = []
        env = CompileEnvironment.current()
        k_index = 0
        for k in index:
            if k is None:
                output_size.append(1)
            elif isinstance(k, int):
                input_size.popleft()
            elif (
                state is not None
                and (tile_info := _get_tile_with_offset_info(k, state, k_index))
                is not None
            ):
                # Tensor marked as tile.index + offset
                input_size.popleft()
                block_id, _ = tile_info
                block_size = env.block_sizes[block_id].var
                if tensor.size(tensor.ndim - len(input_size) - 1) != 1:
                    output_size.append(block_size)
                else:
                    output_size.append(1)
                k_index += 1
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
                k_index += 1
            elif isinstance(k, slice):
                size = input_size.popleft()
                # Handle slices with steps
                slice_size = compute_slice_size(k, size)

                if slice_size != 1:
                    rdim = env.allocate_reduction_dimension(slice_size)
                    output_size.append(rdim.var)
                else:
                    output_size.append(1)
                k_index += 1
            elif isinstance(k, torch.Tensor) and (
                k.ndim == 1 or (len(index) == 1 and tensor.ndim == 1)
            ):
                input_size.popleft()
                output_size.extend(k.size())
                k_index += 1
            else:
                raise exc.InvalidIndexingType(k)
        assert len(input_size) == 0, "invalid subscript"
        return output_size

    @staticmethod
    def _needs_int64(fake_value: torch.Tensor) -> bool:
        storage_offset = fake_value.storage_offset()

        if not isinstance(storage_offset, int):
            return False

        try:
            required = compute_required_storage_length(
                fake_value.shape,
                fake_value.stride(),
                storage_offset,
            )
        except Exception:
            return False

        if not isinstance(required, int):
            return False

        if abs(storage_offset) > torch.iinfo(torch.int32).max:
            return True

        max_offset = required - 1
        return max_offset > torch.iinfo(torch.int32).max

    @staticmethod
    def create(
        state: CodegenState,
        fake_value: torch.Tensor,
        index: list[object],
        extra_mask: ast.AST | None = None,
    ) -> SubscriptIndexing:
        tile_strategy = state.tile_strategy
        output_idx = 0
        index_values = []
        mask_values = {}
        output_size = SubscriptIndexing.compute_shape(fake_value, index, state)
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        if dtype == "tl.int32" and SubscriptIndexing._needs_int64(fake_value):
            raise exc.IndexOffsetOutOfRangeForInt32(env.settings.index_dtype)

        def _is_size_one(size: int | torch.SymInt) -> bool:
            return env.known_equal(size, 1)

        k_index = 0
        for n, k in enumerate(index):
            if k is None:
                output_idx += 1
            elif isinstance(k, int):
                index_values.append(repr(k))
            elif (
                tile_info := _get_tile_with_offset_info(k, state, k_index)
            ) is not None:
                # Tensor marked as tile.index + offset
                block_id, offset = tile_info
                index_var = state.codegen.index_var(block_id)
                offset_expr = state.device_function.literal_expr(offset)
                expand = tile_strategy.expand_str(output_size, output_idx)
                i = len(index_values)
                index_values.append(f"(({index_var}) + {offset_expr}){expand}")
                # Use the same mask as the underlying tile
                if (mask := state.codegen.mask_var(block_id)) and not _is_size_one(
                    fake_value.size(i)
                ):
                    mask_values.setdefault(f"({mask}){expand}")
                output_idx += 1
                k_index += 1
            elif isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    index_var = state.codegen.index_var(origin.origin.block_id)
                    expand = tile_strategy.expand_str(output_size, output_idx)
                    i = len(index_values)
                    index_values.append(f"({index_var}){expand}")
                    if (
                        mask := state.codegen.mask_var(origin.origin.block_id)
                    ) and not _is_size_one(fake_value.size(i)):
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                    k_index += 1
                else:
                    # When the index is a scalar (no BlockSizeOrigin), the corresponding dim is eliminated.
                    val = state.device_function.literal_expr(k)
                    index_values.append(f"({val})")
            elif isinstance(k, slice):
                expand = tile_strategy.expand_str(output_size, output_idx)
                size = fake_value.size(len(index_values))

                # Handle slices with steps
                if k.step is not None and k.step != 1:
                    # For strided slices, we need to generate: start + index * step
                    start = k.start if k.start is not None else 0
                    step = k.step
                    slice_size = compute_slice_size(k, size)

                    if slice_size != 1:
                        rdim = env.allocate_reduction_dimension(slice_size)
                        block_idx = rdim.block_id
                        index_var = state.codegen.index_var(block_idx)
                        # Generate strided index: start + index * step
                        index_values.append(
                            f"({start} + ({index_var}) * {step}){expand}"
                        )
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    else:
                        index_values.append(f"{start}{expand}")
                else:
                    # Full slice or slice without step
                    if not _is_size_one(size):
                        rdim = env.allocate_reduction_dimension(size)
                        block_idx = rdim.block_id
                        index_var = state.codegen.index_var(block_idx)
                        index_values.append(f"({index_var}){expand}")
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    else:
                        index_values.append(f"tl.zeros([1], {dtype}){expand}")
                output_idx += 1
                k_index += 1
            elif isinstance(k, torch.Tensor) and k.ndim == 1:
                expand = tile_strategy.expand_str(output_size, output_idx)
                ast_index = state.ast_args[1]
                assert isinstance(ast_index, (list, tuple))
                assert len(ast_index) == len(index)
                index_var = state.codegen.lift(ast_index[n], prefix="index").id
                index_values.append(f"({index_var}){expand}")
                if (block_idx := env.get_block_id(output_size[output_idx])) is not None:
                    if mask := state.codegen.mask_var(block_idx):
                        mask_values.setdefault(f"({mask}){expand}")
                # Check if this index comes from a padded hl.arange and generate mask
                if (
                    original_length := _get_padded_iota_original_length(state, n)
                ) is not None:
                    mask_values.setdefault(f"({index_var} < {original_length}){expand}")
                output_idx += 1
                k_index += 1
            elif (
                isinstance(k, torch.Tensor) and len(index) == 1 and fake_value.ndim == 1
            ):
                # TODO(jansel): combine this case with the above
                ast_index = state.ast_args[1]
                assert isinstance(ast_index, (list, tuple))
                assert len(ast_index) == 1
                index_var = state.codegen.lift(ast_index[0], prefix="index").id
                index_values.append(index_var)
                output_idx += k.ndim
                for n, s in enumerate(output_size):
                    if (block_idx := env.get_block_id(s)) is not None and (
                        mask := state.codegen.mask_var(block_idx)
                    ):
                        mask_values.setdefault(
                            f"({mask}){tile_strategy.expand_str(output_size, n)}"
                        )
                k_index += 1
            else:
                raise exc.InvalidIndexingType(type(k))
        assert len(output_size) == output_idx
        assert len(index_values) == fake_value.ndim
        index_expr = []
        for i, idx in enumerate(index_values):
            if not _is_size_one(fake_value.size(i)):
                stride = state.device_function.tensor_stride(fake_value, i).name
                index_expr.append(f"{idx} * {stride}")
        if not index_expr:
            shape_str = tile_strategy.shape_str(output_size)
            index_expr.append(f"tl.zeros({shape_str}, {dtype})")

        kwargs = {}
        if extra_mask is not None:
            mask_values.setdefault("{_extra_mask}")
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
        return expr_from_string(f"tl.reshape({{node}}, {shape})", node=node)

    def reshape_store(self, state: CodegenState, node: ast.AST) -> ast.AST:
        if not self.need_reshape(node):
            return node
        shape = state.tile_strategy.shape_str(self.block_shape)
        return expr_from_string(f"tl.reshape({{node}}, {shape})", node=node)

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
        input_sizes = collections.deque(fake_tensor.size())
        k_index = 0
        for k in index:
            input_size = 1 if k is None else input_sizes.popleft()
            # Check for tile+offset tensor first before other checks
            if (
                isinstance(k, torch.Tensor)
                and (tile_info := _get_tile_with_offset_info(k, state, k_index))
                is not None
            ):
                # Tensor marked as tile.index + offset - treat like TileWithOffset
                block_index, _ = tile_info
                try:
                    state.codegen.offset_var(block_index)
                except NotImplementedError:
                    return False
                loop_state = state.codegen.active_device_loops[block_index][-1]
                if isinstance(loop_state, DeviceLoopState):
                    if not loop_state.block_id_to_info[block_index].is_end_matching(
                        input_size
                    ):
                        assert state.fx_node is not None
                        if "masked_value" in state.fx_node.meta:
                            return False
                k_index += 1
            elif isinstance(k, torch.SymInt):
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
                k_index += 1
            elif isinstance(k, torch.Tensor):
                # indirect loads don't work with block_ptr
                return False
        output_shape = SubscriptIndexing.compute_shape(fake_tensor, index, state)
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
        res = BlockedSubscriptIndexing(
            fake_value,
            reshaped_size=SubscriptIndexing.compute_shape(fake_value, index, state),
        )
        env = CompileEnvironment.current()
        k_index = 0
        for k in index:
            if k is None:
                pass  # handled by reshaped_size
            elif isinstance(k, int):
                res.offsets.append(repr(k))
                res.block_shape.append(1)
            elif (
                tile_info := _get_tile_with_offset_info(k, state, k_index)
            ) is not None:
                # Tensor marked as tile.index + offset
                if fake_value.size(len(res.offsets)) != 1:
                    block_id, offset = tile_info
                    offset_var = state.codegen.offset_var(block_id)
                    offset_expr = state.device_function.literal_expr(offset)
                    res.offsets.append(f"({offset_var} + {offset_expr})")
                    res.block_shape.append(env.block_sizes[block_id].var)
                else:
                    res.offsets.append("0")
                    res.block_shape.append(1)
                k_index += 1
            elif isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    if fake_value.size(len(res.offsets)) != 1:
                        res.offsets.append(
                            state.codegen.offset_var(origin.origin.block_id)
                        )
                        res.block_shape.append(k)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
                    k_index += 1
                else:
                    res.offsets.append(state.device_function.literal_expr(k))
                    res.block_shape.append(1)
            elif isinstance(k, slice):
                size = fake_value.size(len(res.offsets))
                # Handle slices with steps
                if k.step is not None and k.step != 1:
                    # Slices with steps are not supported in block_ptr mode
                    raise exc.InvalidIndexingType(
                        f"Strided slices not supported in block_ptr mode: {k}"
                    )
                # Full slice or slice without step
                if size != 1:
                    rdim = env.allocate_reduction_dimension(size)
                    res.offsets.append(state.codegen.offset_var(rdim.block_id))
                    res.block_shape.append(rdim.var)
                else:
                    res.offsets.append("0")
                    res.block_shape.append(1)
                k_index += 1
            else:
                raise exc.InvalidIndexingType(k)
        res.validate()
        return res
