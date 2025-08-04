from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch

from .. import exc
from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .host_function import HostFunction
from .tile_strategy import DeviceLoopState
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .device_function import TensorDescriptorArg
    from .inductor_lowering import CodegenState


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
        if isinstance(value, ast.Name) and hasattr(state.device_function, '_tensor_args'):
            # Check if this name corresponds to a tensor argument
            for tensor, tensor_arg in state.device_function._tensor_args.items():
                if tensor_arg.name == value.id:
                    # This is a tensor value, we need to load from it
                    # Get the shape of the slice we're storing to
                    output_shape = SubscriptIndexing.compute_shape(fake_tensor, subscript)
                    if len(output_shape) == 1 and tensor.ndim == 1:
                        # Load the entire 1D tensor
                        value_indexing = SubscriptIndexing.create(state, tensor, [slice(None)], None)
                        value = expr_from_string(
                            f"tl.load({value.id} + offset, mask)",
                            offset=value_indexing.index_expr,
                            mask=value_indexing.mask_expr,
                        )
                    break
        
        return expr_from_string(
            f"tl.store({name} + offset, value, mask)",
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
    ) -> ast.AST:
        if not BlockedSubscriptIndexing.is_supported(
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
        if not BlockedSubscriptIndexing.is_supported(
            state, fake_tensor, subscript, extra_mask
        ):
            return PointerIndexingStrategy().codegen_store(
                state, fake_tensor, subscript, value, extra_mask
            )
        assert extra_mask is None
        indexing = BlockedSubscriptIndexing.create(state, fake_tensor, subscript)
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
            elif k is Ellipsis:
                # Ellipsis is not supported in tensor descriptor mode
                return False
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


class SubscriptIndexing(NamedTuple):
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )

    @staticmethod
    def compute_shape(
        tensor: torch.Tensor, index: list[object]
    ) -> list[int | torch.SymInt]:
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(index, (list, tuple)), index
        input_size = collections.deque(tensor.size())
        output_size = []
        env = CompileEnvironment.current()
        for i, k in enumerate(index):
            if k is None:
                output_size.append(1)
            elif k is Ellipsis:
                # Ellipsis expands to consume all remaining dims except those after it
                remaining_indices = len(index) - i - 1
                ellipsis_dims = len(input_size) - remaining_indices
                for _ in range(ellipsis_dims):
                    size = input_size.popleft()
                    if size != 1:
                        rdim = env.allocate_reduction_dimension(size)
                        output_size.append(rdim.var)
                    else:
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
                if k.step is not None and k.step != 1:
                    # Calculate size based on step
                    start = k.start if k.start is not None else 0
                    stop = k.stop if k.stop is not None else size
                    step = k.step
                    slice_size = (stop - start + step - 1) // step
                else:
                    # Full slice or slice without step
                    slice_size = size
                
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
        # For partial indexing, append remaining dimensions to output
        while input_size:
            size = input_size.popleft()
            if size != 1:
                rdim = env.allocate_reduction_dimension(size)
                output_size.append(rdim.var)
            else:
                output_size.append(1)
        return output_size

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
        output_size = SubscriptIndexing.compute_shape(fake_value, index)
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        for n, k in enumerate(index):
            if k is None:
                output_idx += 1
            elif k is Ellipsis:
                # Ellipsis expands to handle remaining dimensions
                remaining_indices = len(index) - n - 1
                ellipsis_dims = fake_value.ndim - len(index_values) - remaining_indices
                for dim_offset in range(ellipsis_dims):
                    expand = tile_strategy.expand_str(output_size, output_idx)
                    size = fake_value.size(len(index_values))
                    if size != 1:
                        rdim = env.allocate_reduction_dimension(size)
                        block_idx = rdim.block_id
                        index_var = state.codegen.index_var(block_idx)
                        index_values.append(f"({index_var}){expand}")
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    else:
                        index_values.append(f"tl.zeros([1], {dtype}){expand}")
                    output_idx += 1
            elif isinstance(k, int):
                # Normalize negative indices
                if k < 0:
                    dim_idx = len(index_values)
                    dim_size = fake_value.size(dim_idx)
                    # Handle both concrete and symbolic dimension sizes
                    if isinstance(dim_size, int):
                        normalized_k = k + dim_size
                        index_values.append(repr(normalized_k))
                    else:
                        # For symbolic dimensions, we need to generate the proper expression
                        # The state.codegen is a GenerateAST instance which has device_function
                        sympy_expr = dim_size._sympy_() + k
                        index_values.append(f"({state.codegen.device_function.user_sympy_expr(sympy_expr)})")
                else:
                    index_values.append(repr(k))
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
                    ) and fake_value.size(i) != 1:
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
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
                    stop = k.stop if k.stop is not None else size
                    step = k.step
                    slice_size = (stop - start + step - 1) // step
                    
                    if slice_size != 1:
                        rdim = env.allocate_reduction_dimension(slice_size)
                        block_idx = rdim.block_id
                        index_var = state.codegen.index_var(block_idx)
                        # Generate strided index: start + index * step
                        index_values.append(f"({start} + ({index_var}) * {step}){expand}")
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    else:
                        index_values.append(f"{start}{expand}")
                else:
                    # Full slice or slice without step
                    if size != 1:
                        rdim = env.allocate_reduction_dimension(size)
                        block_idx = rdim.block_id
                        index_var = state.codegen.index_var(block_idx)
                        index_values.append(f"({index_var}){expand}")
                        if mask := state.codegen.mask_var(block_idx):
                            mask_values.setdefault(f"({mask}){expand}")
                    else:
                        index_values.append(f"tl.zeros([1], {dtype}){expand}")
                output_idx += 1
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
                output_idx += 1
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
            else:
                raise exc.InvalidIndexingType(type(k))
        
        # Handle remaining dimensions for partial indexing
        while len(index_values) < fake_value.ndim:
            expand = tile_strategy.expand_str(output_size, output_idx)
            size = fake_value.size(len(index_values))
            if size != 1:
                rdim = env.allocate_reduction_dimension(size)
                block_idx = rdim.block_id
                index_var = state.codegen.index_var(block_idx)
                index_values.append(f"({index_var}){expand}")
                if mask := state.codegen.mask_var(block_idx):
                    mask_values.setdefault(f"({mask}){expand}")
            else:
                index_values.append(f"tl.zeros([1], {dtype}){expand}")
            output_idx += 1
            
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
        input_sizes = collections.deque(fake_tensor.size())
        for n, k in enumerate(index):
            if k is None:
                input_size = 1
            elif k is Ellipsis:
                # Skip appropriate number of dimensions for ellipsis
                remaining_indices = len(index) - n - 1
                ellipsis_dims = len(input_sizes) - remaining_indices
                for _ in range(ellipsis_dims):
                    input_sizes.popleft()
                continue
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
        res = BlockedSubscriptIndexing(
            fake_value,
            reshaped_size=SubscriptIndexing.compute_shape(fake_value, index),
        )
        for n, k in enumerate(index):
            if k is None:
                pass  # handled by reshaped_size
            elif k is Ellipsis:
                # Ellipsis expands to handle remaining dimensions
                remaining_indices = len(index) - n - 1
                ellipsis_dims = fake_value.ndim - len(res.offsets) - remaining_indices
                for _ in range(ellipsis_dims):
                    size = fake_value.size(len(res.offsets))
                    if size != 1:
                        env = CompileEnvironment.current()
                        rdim = env.allocate_reduction_dimension(size)
                        res.offsets.append(state.codegen.offset_var(rdim.block_id))
                        res.block_shape.append(rdim.var)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
            elif isinstance(k, int):
                # Normalize negative indices
                if k < 0:
                    dim_idx = len(res.offsets)
                    dim_size = fake_value.size(dim_idx)
                    # Handle both concrete and symbolic dimension sizes
                    if isinstance(dim_size, int):
                        normalized_k = k + dim_size
                        res.offsets.append(repr(normalized_k))
                    else:
                        # For symbolic dimensions, we need to generate the proper expression
                        # The state.codegen is a GenerateAST instance which has device_function
                        sympy_expr = dim_size._sympy_() + k
                        res.offsets.append(f"({state.codegen.device_function.user_sympy_expr(sympy_expr)})")
                else:
                    res.offsets.append(repr(k))
                res.block_shape.append(1)
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
                else:
                    res.offsets.append(state.device_function.literal_expr(k))
                    res.block_shape.append(1)
            elif isinstance(k, slice):
                size = fake_value.size(len(res.offsets))
                # Handle slices with steps
                if k.step is not None and k.step != 1:
                    # Slices with steps are not supported in block_ptr mode
                    raise exc.InvalidIndexingType(f"Strided slices not supported in block_ptr mode: {k}")
                else:
                    # Full slice or slice without step
                    if size != 1:
                        env = CompileEnvironment.current()
                        rdim = env.allocate_reduction_dimension(size)
                        res.offsets.append(state.codegen.offset_var(rdim.block_id))
                        res.block_shape.append(rdim.var)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
            else:
                raise exc.InvalidIndexingType(k)
        res.validate()
        return res
