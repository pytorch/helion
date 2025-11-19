from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

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
            and index_node.target == torch.ops.prims.iota.default
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
            # pyrefly: ignore [bad-argument-type]
            ev=eviction_policy,
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
                # pyrefly: ignore [bad-argument-type]
                ev=eviction_policy,
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
            # pyrefly: ignore [bad-argument-type]
            ev=eviction_policy,
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


@dataclasses.dataclass
class _SubscriptIndexingContext:
    """Context object to hold state during index processing."""
    state: CodegenState
    fake_value: torch.Tensor
    index: list[object]
    env: CompileEnvironment
    dtype: str

    # Output tracking
    output_idx: int = 0
    index_values: list[str] = dataclasses.field(default_factory=list)
    mask_values: dict[str, None] = dataclasses.field(default_factory=dict)

    # Computed values (initialized in __post_init__)
    output_size: list[int | torch.SymInt] = dataclasses.field(init=False)
    tile_strategy: object = dataclasses.field(init=False)
    all_tensors: list[torch.Tensor] = dataclasses.field(init=False)
    tensor_shapes: list[list[int | torch.SymInt]] = dataclasses.field(init=False)
    broadcast_shape: list[int | torch.SymInt] | None = dataclasses.field(init=False)

    # Tensor indexing state
    first_tensor_idx: int = 0
    tensor_count: int = 0
    k_index: int = 0

    def __post_init__(self) -> None:
        self.output_size = SubscriptIndexing.compute_shape(
            self.fake_value, self.index, self.state
        )
        self.tile_strategy = self.state.tile_strategy
        self.all_tensors = [
            cast(torch.Tensor, k) for k in self.index
            if isinstance(k, torch.Tensor)
        ]
        self.tensor_shapes = [list(t.size()) for t in self.all_tensors]
        self.broadcast_shape = (
            self.env.tensor_indexer_broadcast_shape(self.all_tensors)
            if self.all_tensors else None
        )

    def is_size_one(self, size: int | torch.SymInt) -> bool:
        """Check if a size is known to be one."""
        return self.env.known_equal(size, 1)


@dataclasses.dataclass
class _TensorShapeInfo:
    """Information about tensor shape for broadcasting."""
    expand_pos: int
    width: int


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
        output_size: list[int | torch.SymInt] = []
        env = CompileEnvironment.current()

        # Get broadcast shape for tensor indexers
        tensors = [cast(torch.Tensor, k) for k in index if isinstance(k, torch.Tensor)]
        broadcast_shape = env.tensor_indexer_broadcast_shape(tensors) if tensors else None
        use_broadcast_once = broadcast_shape is not None
        added_broadcast_shape = False

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
                slice_size = compute_slice_size(k, size)
                if slice_size != 1:
                    rdim = env.allocate_reduction_dimension(slice_size)
                    output_size.append(rdim.var)
                else:
                    output_size.append(1)
                k_index += 1
            elif isinstance(k, torch.Tensor):
                if use_broadcast_once:
                    input_size.popleft()
                    if not added_broadcast_shape:
                        output_size.extend(broadcast_shape)  # pyright: ignore[reportArgumentType]
                        added_broadcast_shape = True
                else:
                    base_dim = input_size.popleft()
                    output_size.extend(env.get_indexer_output_dims(k, base_dim))
                k_index += 1
            else:
                raise exc.InvalidIndexingType(k)
        # Advanced indexing might not consume all dimensions
        # Add any remaining dimensions from the input
        output_size.extend(input_size)
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
        """Create a SubscriptIndexing instance for the given tensor and index."""
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()

        if dtype == "tl.int32" and SubscriptIndexing._needs_int64(fake_value):
            raise exc.IndexOffsetOutOfRangeForInt32(env.index_dtype)

        # Initialize context for index processing
        context = _SubscriptIndexingContext(
            state=state,
            fake_value=fake_value,
            index=index,
            env=env,
            dtype=dtype
        )

        # Process each index element
        for position, index_elem in enumerate(index):
            SubscriptIndexing._process_index_element(context, position, index_elem)

        # Validate and build final expressions
        assert len(context.output_size) == context.output_idx
        assert len(context.index_values) == fake_value.ndim

        return SubscriptIndexing._build_final_expressions(context, extra_mask)

    @staticmethod
    def _process_index_element(
        ctx: _SubscriptIndexingContext,
        position: int,
        index_elem: object
    ) -> None:
        """Process a single index element and update context."""
        if index_elem is None:
            ctx.output_idx += 1
        elif isinstance(index_elem, int):
            SubscriptIndexing._process_int_index(ctx, index_elem)
        elif (tile_info := _get_tile_with_offset_info(
            index_elem, ctx.state, ctx.k_index
        )) is not None:
            SubscriptIndexing._process_tile_with_offset(ctx, tile_info)
        elif isinstance(index_elem, torch.SymInt):
            SubscriptIndexing._process_symint_index(ctx, index_elem)
        elif isinstance(index_elem, slice):
            SubscriptIndexing._process_slice_index(ctx, index_elem)
        elif isinstance(index_elem, torch.Tensor):
            SubscriptIndexing._process_tensor_index(ctx, position, index_elem)
        else:
            raise exc.InvalidIndexingType(type(index_elem))

    @staticmethod
    def _process_int_index(ctx: _SubscriptIndexingContext, value: int) -> None:
        """Process an integer index."""
        ctx.index_values.append(repr(value))

    @staticmethod
    def _add_block_index_and_mask(
        ctx: _SubscriptIndexingContext,
        block_id: int,
        extra_offset: str = "",
        position_override: int | None = None
    ) -> str:
        """Add block index variable and associated mask if needed.

        Returns:
            The expand string used.
        """
        index_var = ctx.state.codegen.index_var(block_id)
        pos = position_override if position_override is not None else ctx.output_idx
        expand = ctx.tile_strategy.expand_str(ctx.output_size, pos)

        # Build index expression
        index_expr = f"({index_var})"
        if extra_offset:
            index_expr = f"({index_expr} + {extra_offset})"
        ctx.index_values.append(f"{index_expr}{expand}")

        # Add mask if needed
        SubscriptIndexing._add_mask_if_needed(ctx, block_id, expand, len(ctx.index_values) - 1)
        return expand

    @staticmethod
    def _add_mask_if_needed(
        ctx: _SubscriptIndexingContext,
        block_id: int | None,
        expand: str,
        dim_index: int | None = None
    ) -> None:
        """Add mask for a block if it exists and dimension is not size one."""
        if block_id is None:
            return

        mask = ctx.state.codegen.mask_var(block_id)
        if not mask:
            return

        # Check if dimension is size one (can be skipped)
        if dim_index is not None and ctx.is_size_one(ctx.fake_value.size(dim_index)):
            return

        ctx.mask_values.setdefault(f"({mask}){expand}")

    @staticmethod
    def _get_symint_block_origin(symint: torch.SymInt) -> BlockSizeOrigin | None:
        """Extract BlockSizeOrigin from a SymInt if it has one."""
        symbol = symint._sympy_()
        if not isinstance(symbol, sympy.Symbol):
            return None
        origin = HostFunction.current().expr_to_origin.get(symbol)
        if origin and isinstance(origin.origin, BlockSizeOrigin):
            return origin.origin
        return None

    @staticmethod
    def _process_tile_with_offset(
        ctx: _SubscriptIndexingContext,
        tile_info: tuple[int, int | torch.SymInt]
    ) -> None:
        """Process a tensor marked as tile.index + offset."""
        block_id, offset = tile_info
        offset_expr = ctx.state.device_function.literal_expr(offset)
        SubscriptIndexing._add_block_index_and_mask(ctx, block_id, offset_expr)
        ctx.output_idx += 1
        ctx.k_index += 1

    @staticmethod
    def _process_symint_index(ctx: _SubscriptIndexingContext, symint: torch.SymInt) -> None:
        """Process a SymInt index."""
        origin = SubscriptIndexing._get_symint_block_origin(symint)

        if origin:
            # Handle block size origin
            SubscriptIndexing._add_block_index_and_mask(ctx, origin.block_id)
            ctx.output_idx += 1
            ctx.k_index += 1
        else:
            # Scalar index - dimension is eliminated
            val = ctx.state.device_function.literal_expr(symint)
            ctx.index_values.append(f"({val})")

    @staticmethod
    def _process_slice_index(ctx: _SubscriptIndexingContext, slice_obj: slice) -> None:
        """Process a slice index."""
        size = ctx.fake_value.size(len(ctx.index_values))
        slice_size = compute_slice_size(slice_obj, size)

        if slice_obj.step is not None and slice_obj.step != 1:
            # Handle strided slices
            start = slice_obj.start if slice_obj.start is not None else 0
            if slice_size != 1:
                rdim = ctx.env.allocate_reduction_dimension(slice_size)
                expand = ctx.tile_strategy.expand_str(ctx.output_size, ctx.output_idx)
                index_var = ctx.state.codegen.index_var(rdim.block_id)
                # Generate strided index: start + index * step
                ctx.index_values.append(f"({start} + ({index_var}) * {slice_obj.step}){expand}")
                SubscriptIndexing._add_mask_if_needed(ctx, rdim.block_id, expand, len(ctx.index_values) - 1)
            else:
                expand = ctx.tile_strategy.expand_str(ctx.output_size, ctx.output_idx)
                ctx.index_values.append(f"{start}{expand}")
        else:
            # Handle regular slices
            if not ctx.is_size_one(size):
                rdim = ctx.env.allocate_reduction_dimension(size)
                SubscriptIndexing._add_block_index_and_mask(ctx, rdim.block_id)
            else:
                expand = ctx.tile_strategy.expand_str(ctx.output_size, ctx.output_idx)
                ctx.index_values.append(f"tl.zeros([1], {ctx.dtype}){expand}")

        ctx.output_idx += 1
        ctx.k_index += 1

    @staticmethod
    def _process_tensor_index(
        ctx: _SubscriptIndexingContext,
        position: int,
        tensor: torch.Tensor
    ) -> None:
        """Process a tensor index with broadcasting support."""
        # Get the index variable from AST
        ast_index = ctx.state.ast_args[1]
        assert isinstance(ast_index, (list, tuple))
        index_var = ctx.state.codegen.lift(ast_index[position], prefix="index").id

        # Try special case for 2D Cartesian product first
        if SubscriptIndexing._try_2d_cartesian_product(ctx, index_var, position):
            return

        # Handle general tensor indexing
        if ctx.broadcast_shape:
            SubscriptIndexing._process_broadcast_tensor(ctx, tensor, index_var)
        else:
            SubscriptIndexing._process_simple_tensor(ctx, tensor, index_var)

        ctx.tensor_count += 1
        ctx.k_index += 1

    @staticmethod
    def _try_2d_cartesian_product(
        ctx: _SubscriptIndexingContext,
        index_var: str,
        position: int
    ) -> bool:
        """Try to handle 2D Cartesian product special case."""
        # Only apply this special case when we have exactly 2 tensor indices
        # with broadcast shape of length 2, regardless of other index types
        if not (ctx.broadcast_shape and
                len(ctx.all_tensors) == 2 and
                len(ctx.broadcast_shape) == 2):
            return False

        # Check if all tensors are effectively 1D
        all_1d = all(
            len(s) == 1 or sum(1 for d in s if ctx.env.size_hint(d) != 1) <= 1
            for s in ctx.tensor_shapes
        )
        if not all_1d:
            return False

        original_length = _get_padded_iota_original_length(ctx.state, position)

        if ctx.tensor_count == 0:
            ctx.index_values.append(f"({index_var})[:, None]")
            ctx.first_tensor_idx = ctx.output_idx
            ctx.output_idx += 2
            if original_length is not None:
                ctx.mask_values.setdefault(f"(({index_var} < {original_length})[:, None])")
        else:
            ctx.index_values.append(f"({index_var})[None, :]")
            if original_length is not None:
                ctx.mask_values.setdefault(f"(({index_var} < {original_length})[None, :])")

        ctx.tensor_count += 1
        ctx.k_index += 1
        return True

    @staticmethod
    def _process_broadcast_tensor(
        ctx: _SubscriptIndexingContext,
        tensor: torch.Tensor,
        index_var: str
    ) -> None:
        """Process tensor indexing with broadcasting."""
        if ctx.tensor_count == 0:
            ctx.first_tensor_idx = ctx.output_idx
            ctx.output_idx += len(ctx.broadcast_shape)

        # Determine dimensions and positioning
        shape_info = SubscriptIndexing._get_tensor_shape_info(ctx, tensor)

        if shape_info.width <= 1:
            SubscriptIndexing._process_single_dim_broadcast(
                ctx, tensor, index_var, shape_info.expand_pos
            )
        else:
            SubscriptIndexing._process_multi_dim_broadcast(
                ctx, index_var, shape_info.expand_pos, shape_info.width
            )

    @staticmethod
    def _process_tensor_with_tile_origin(
        ctx: _SubscriptIndexingContext,
        tensor: torch.Tensor,
        index_var: str,
        expand_pos: int | None = None
    ) -> None:
        """Process tensor indexing with potential tile origin block ID."""
        pos = expand_pos if expand_pos is not None else ctx.output_idx
        expand = ctx.tile_strategy.expand_str(ctx.output_size, pos)
        tile_origin_block_id = ctx.env.get_tile_index_tensor_block_id(tensor)

        if tile_origin_block_id is not None:
            index_var = ctx.state.codegen.index_var(tile_origin_block_id)
            ctx.index_values.append(f"({index_var}){expand}")
            SubscriptIndexing._add_mask_if_needed(ctx, tile_origin_block_id, expand, len(ctx.index_values) - 1)
        else:
            ctx.index_values.append(f"({index_var}){expand}")
            block_idx = ctx.env.get_block_id(ctx.output_size[pos]) if pos < len(ctx.output_size) else None
            SubscriptIndexing._add_mask_if_needed(ctx, block_idx, expand, len(ctx.index_values) - 1)

    @staticmethod
    def _process_simple_tensor(
        ctx: _SubscriptIndexingContext,
        tensor: torch.Tensor,
        index_var: str
    ) -> None:
        """Process simple tensor indexing without broadcasting."""
        SubscriptIndexing._process_tensor_with_tile_origin(ctx, tensor, index_var)
        ctx.output_idx += tensor.ndim

    @staticmethod
    def _get_tensor_shape_info(
        ctx: _SubscriptIndexingContext,
        tensor: torch.Tensor
    ) -> _TensorShapeInfo:
        """Get shape information for tensor broadcasting."""
        shape_size = (len(ctx.tensor_shapes[ctx.tensor_count])
                     if ctx.tensor_count < len(ctx.tensor_shapes) else 1)

        # Check if tensor has more than 1 non-singleton dimension
        non_bcast_dims = (
            sum(1 for d in ctx.tensor_shapes[ctx.tensor_count]
                if ctx.env.size_hint(d) != 1)
            if ctx.tensor_count < len(ctx.tensor_shapes) else 0
        )
        is_single_dim = non_bcast_dims <= 1

        # Calculate positioning
        offset = max(0, len(ctx.broadcast_shape) - shape_size)
        if is_single_dim and shape_size > 0:
            non_one_positions = [
                i for i, d in enumerate(ctx.tensor_shapes[ctx.tensor_count])
                if ctx.env.size_hint(d) != 1
            ]
            rel_pos = non_one_positions[0] if non_one_positions else (shape_size - 1)
            expand_pos = ctx.first_tensor_idx + offset + rel_pos
        else:
            expand_pos = ctx.first_tensor_idx + offset

        expand_pos = max(0, min(expand_pos, len(ctx.output_size) - 1)) if ctx.output_size else 0
        width = 1 if is_single_dim else min(tensor.ndim, max(0, len(ctx.output_size) - expand_pos))

        return _TensorShapeInfo(expand_pos, width)

    @staticmethod
    def _process_single_dim_broadcast(
        ctx: _SubscriptIndexingContext,
        tensor: torch.Tensor,
        index_var: str,
        expand_pos: int
    ) -> None:
        """Process single-dimensional broadcast tensor."""
        SubscriptIndexing._process_tensor_with_tile_origin(ctx, tensor, index_var, expand_pos)

    @staticmethod
    def _process_multi_dim_broadcast(
        ctx: _SubscriptIndexingContext,
        index_var: str,
        expand_pos: int,
        width: int
    ) -> None:
        """Process multi-dimensional broadcast tensor."""
        positions = [expand_pos + d for d in range(width)]

        # Build bracket for multi-dimensional expansion
        bracket = SubscriptIndexing._build_multi_dim_bracket(ctx, positions)
        ctx.index_values.append(f"({index_var}){bracket}")

        # Add masks for each position
        for pos in positions:
            block_idx = ctx.env.get_block_id(ctx.output_size[pos]) if pos < len(ctx.output_size) else None
            if block_idx is not None:
                expand = ctx.tile_strategy.expand_str(ctx.output_size, pos)
                SubscriptIndexing._add_mask_if_needed(ctx, block_idx, expand)

    @staticmethod
    def _build_multi_dim_bracket(
        ctx: _SubscriptIndexingContext,
        positions: list[int]
    ) -> str:
        """Build bracket string for multi-dimensional expansion."""
        tokens: list[str] | None = None

        for pos in positions:
            expand = ctx.tile_strategy.expand_str(ctx.output_size, pos)
            if expand == "":
                current = [":"]
            else:
                assert expand.startswith("[") and expand.endswith("]"), expand
                current = expand[1:-1].split(", ") if len(expand) > 2 else []

            if tokens is None:
                tokens = current
            elif current:
                tokens = [
                    ":" if (a == ":" or b == ":") else "None"
                    for a, b in zip(tokens, current, strict=True)
                ]

        return f"[{', '.join(tokens)}]" if tokens and not all(t == ":" for t in tokens) else ""

    @staticmethod
    def _build_final_expressions(
        ctx: _SubscriptIndexingContext,
        extra_mask: ast.AST | None
    ) -> SubscriptIndexing:
        """Build the final index and mask expressions."""
        # Build index expression
        index_expr = []
        for i, idx in enumerate(ctx.index_values):
            if not ctx.is_size_one(ctx.fake_value.size(i)):
                stride = ctx.state.device_function.tensor_stride(ctx.fake_value, i).name
                index_expr.append(f"{idx} * {stride}")

        if not index_expr:
            shape_str = ctx.tile_strategy.shape_str(ctx.output_size)
            index_expr.append(f"tl.zeros({shape_str}, {ctx.dtype})")

        # Build mask expression
        kwargs = {}
        if extra_mask is not None:
            ctx.mask_values.setdefault("{_extra_mask}")
            kwargs["_extra_mask"] = extra_mask

        return SubscriptIndexing(
            expr_from_string("+".join(index_expr)),
            expr_from_string("&".join(ctx.mask_values) or "None", **kwargs),
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
                if not BlockedSubscriptIndexing._check_block_index_support(
                    state, block_index, input_size
                ):
                    return False
                k_index += 1
            elif isinstance(k, torch.SymInt):
                origin = SubscriptIndexing._get_symint_block_origin(k)
                if origin:
                    if not BlockedSubscriptIndexing._check_block_index_support(
                        state, origin.block_id, input_size
                    ):
                        return False
                k_index += 1
            elif isinstance(k, torch.Tensor):
                # indirect loads don't work with block_ptr
                return False
        output_shape = SubscriptIndexing.compute_shape(fake_tensor, index, state)
        return len(output_shape) != 0

    @staticmethod
    def _check_block_index_support(
        state: CodegenState, block_index: int, input_size: int | torch.SymInt
    ) -> bool:
        """Check if a block index is supported for block_ptr."""
        try:
            state.codegen.offset_var(block_index)
        except NotImplementedError:
            return False
        loop_state = state.codegen.active_device_loops[block_index][-1]
        if isinstance(loop_state, DeviceLoopState):
            if not loop_state.block_id_to_info[block_index].is_end_matching(input_size):
                assert state.fx_node is not None
                if "masked_value" in state.fx_node.meta:
                    # TODO(jansel): in this case we should be able to lower to block_ptr+tl.where
                    # see test/test_loops.py::TestLoops::test_data_dependent_bounds2
                    return False
        return True

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
                origin = SubscriptIndexing._get_symint_block_origin(k)
                if origin:
                    if fake_value.size(len(res.offsets)) != 1:
                        res.offsets.append(
                            state.codegen.offset_var(origin.block_id)
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
