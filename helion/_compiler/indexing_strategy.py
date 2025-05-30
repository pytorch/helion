from __future__ import annotations

import ast
import collections
import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch

from .. import exc
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .tile_strategy import TileStrategy
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
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
        extra = ", other=0" if indexing.has_mask() else ""
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
                f"{indexing.tensor_descriptor(state)}.load({indexing.offsets_str()})"
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
            f"{indexing.tensor_descriptor(state)}.store({indexing.offsets_str()}, value)",
            value=indexing.reshape_store(state, value),
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
        assert isinstance(index, (list, tuple))
        input_size = collections.deque(tensor.size())
        output_size = []
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
                        if (
                            CompileEnvironment.current()
                            .block_sizes[origin.origin.block_size_idx]
                            .is_grid()
                        ):
                            pass
                        elif tensor.size(tensor.ndim - len(input_size) - 1) != 1:
                            output_size.append(k)
                        else:
                            output_size.append(1)
            elif isinstance(k, slice) and str(k) == "slice(None, None, None)":
                size = input_size.popleft()
                if size != 1:
                    rdim = CompileEnvironment.current().allocate_reduction_dimension(
                        size
                    )
                    output_size.append(rdim.var)
                else:
                    output_size.append(1)
            elif isinstance(k, torch.Tensor) and k.ndim == 1:
                input_size.popleft()
                output_size.append(k.size(0))
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
        tile_strategy = state.tile_strategy
        output_idx = 0
        index_values = []
        mask_values = {}
        output_size = SubscriptIndexing.compute_shape(fake_value, index)
        dtype = CompileEnvironment.current().triton_index_type()
        first_non_grid_index = 0
        for n, k in enumerate(index):
            if k is None:
                output_idx += 1
            elif isinstance(k, int):
                index_values.append(repr(k))
            elif isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    if (
                        CompileEnvironment.current()
                        .block_sizes[origin.origin.block_size_idx]
                        .is_grid()
                    ):
                        first_non_grid_index = n + 1
                        expand = tile_strategy.expand_str(output_size, output_idx)
                    else:
                        expand = tile_strategy.expand_str(
                            output_size, output_idx - first_non_grid_index
                        )
                    index_var = state.codegen.index_var(origin.origin.block_size_idx)
                    i = len(index_values)
                    index_values.append(f"({index_var}){expand}")
                    if (
                        mask := state.codegen.mask_var(origin.origin.block_size_idx)
                    ) and fake_value.size(i) != 1:
                        mask_values.setdefault(f"({mask}){expand}")
                    output_idx += 1
                else:
                    expand = tile_strategy.expand_str(
                        output_size, output_idx - first_non_grid_index
                    )
                    val = state.device_function.literal_expr(k)
                    index_values.append(f"tl.full([1], {val}, {dtype}){expand}")
            elif isinstance(k, slice) and str(k) == "slice(None, None, None)":
                expand = tile_strategy.expand_str(
                    output_size, output_idx - first_non_grid_index
                )
                size = fake_value.size(len(index_values))
                if size != 1:
                    env = CompileEnvironment.current()
                    rdim = env.allocate_reduction_dimension(size)
                    block_idx = rdim.block_size_idx
                    index_var = state.codegen.index_var(block_idx)
                    index_values.append(f"({index_var}){expand}")
                    if mask := state.codegen.mask_var(block_idx):
                        mask_values.setdefault(f"({mask}){expand}")
                else:
                    index_values.append(f"tl.zeros([1], {dtype}){expand}")
                output_idx += 1
            elif isinstance(k, torch.Tensor) and k.ndim == 1:
                expand = tile_strategy.expand_str(
                    output_size, output_idx - first_non_grid_index
                )
                ast_index = state.ast_args[1]
                assert isinstance(ast_index, (list, tuple))
                assert len(ast_index) == len(index)
                index_var = state.codegen.lift(ast_index[n]).id
                index_values.append(f"({index_var}){expand}")
                if (
                    block_idx := TileStrategy.get_block_index(
                        output_size[output_idx - first_non_grid_index]
                    )
                ) is not None:
                    if mask := state.codegen.mask_var(block_idx):
                        mask_values.setdefault(f"({mask}){expand}")
                output_idx += 1
            else:
                raise exc.InvalidIndexingType(type(k))
        assert len(output_size) == output_idx - first_non_grid_index
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

    def offsets_str(self) -> str:
        return f"[{', '.join(self.offsets)}]"

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

    def need_reshape(self) -> bool:
        if len(self.reshaped_size) != len(self.block_shape):
            return True
        env = CompileEnvironment.current()
        for a, b in zip(self.reshaped_size, self.block_shape, strict=True):
            if not env.known_equal(a, b):
                return True
        return False

    def reshape_load(self, state: CodegenState, node: ast.AST) -> ast.AST:
        if not self.need_reshape():
            return node
        shape = state.tile_strategy.shape_str(self.reshaped_size)
        return expr_from_string(f"tl.reshape(node, {shape})", node=node)

    def reshape_store(self, state: CodegenState, node: ast.AST) -> ast.AST:
        if not self.need_reshape():
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
            # TODO(jansel): block_ptr with extra_mask
            return False
        for k in index:
            if isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = None
                if isinstance(symbol, sympy.Symbol):
                    origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    try:
                        state.codegen.offset_var(origin.origin.block_size_idx)
                    except NotImplementedError:
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
        for k in index:
            if k is None:
                pass  # handled by reshaped_size
            elif isinstance(k, int):
                res.offsets.append(repr(k))
                res.block_shape.append(1)
            elif isinstance(k, torch.SymInt):
                symbol = k._sympy_()
                origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin and isinstance(origin.origin, BlockSizeOrigin):
                    if fake_value.size(len(res.offsets)) != 1:
                        res.offsets.append(
                            state.codegen.offset_var(origin.origin.block_size_idx)
                        )
                        res.block_shape.append(k)
                    else:
                        res.offsets.append("0")
                        res.block_shape.append(1)
                else:
                    res.offsets.append(state.device_function.literal_expr(k))
                    res.block_shape.append(1)
            elif isinstance(k, slice) and str(k) == "slice(None, None, None)":
                size = fake_value.size(len(res.offsets))
                if size != 1:
                    env = CompileEnvironment.current()
                    rdim = env.allocate_reduction_dimension(size)
                    res.offsets.append(state.codegen.offset_var(rdim.block_size_idx))
                    res.block_shape.append(rdim.var)
                else:
                    res.offsets.append("0")
                    res.block_shape.append(1)
            else:
                raise exc.InvalidIndexingType(k)
        res.validate()
        return res
