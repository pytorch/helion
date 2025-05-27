from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
import weakref

import sympy
import torch

from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import LoopSpecBlockSizeSource
from .host_function import HostFunction
from .program_id import GridProgramIDs
from .program_id import L2GroupingProgramIDs
from .program_id import ProgramID
from .program_id import ProgramIDs
from .program_id import VirtualProgramIDs
from .variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..autotuner.config_spec import BlockSizeSpec
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy

    @property
    def block_indices(self) -> list[int]:
        return self.strategy.block_indices


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)


class DeviceGridState(DeviceLoopOrGridState):
    pass


class PersistentReductionState(DeviceLoopOrGridState):
    pass


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_indices: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_indices = block_indices
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_indices
        }
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"offset_{block_idx}", dce=True)
            for block_idx in block_indices
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> None:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    @classmethod
    def get_block_index(cls, size: int | torch.SymInt | sympy.Expr) -> int | None:
        if isinstance(size, torch.SymInt):
            return cls.get_block_index(size._sympy_())
        if isinstance(size, sympy.Symbol):
            origin_info = HostFunction.current().expr_to_origin.get(size)
            if origin_info is not None and isinstance(
                origin_info.origin,
                BlockSizeOrigin,
            ):
                return origin_info.origin.block_size_idx
        return None


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_indices=block_indices,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_indices: list[_T]) -> list[_T]:
        if len(block_indices) <= 1:
            return block_indices
        order = self.loop_order
        assert len(order) == len(block_indices), (
            f"Invalid order length: {len(order)} != {len(block_indices)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_indices[i] for i in reversed(order)]

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_indices, block_size, loop_order)
        env = CompileEnvironment.current()
        if env.known_multiple(
            functools.reduce(
                operator.mul, [env.block_sizes[i].numel for i in block_indices]
            ),
            block_size,
        ):
            self._mask_var: str | None = None
        else:
            self._mask_var = self.new_var("mask", dce=True)

        key = (*self.block_indices,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_indices:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_indices))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_indices)]

    def _codegen_common(
        self, state: CodegenState
    ) -> tuple[str, str, sympy.Expr, list[ast.AST]]:
        block_indices = self.block_indices
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        device_function = state.device_function
        offsets_var = self.new_var("offsets", dce=True)
        block_size_var = self.block_size_var(-1)
        statements = []
        if state.device_function.constexpr_arg(block_size_var):
            block_size_str = HostFunction.current().literal_expr(self.block_size)
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {block_size_str}")
            )
        for i, block_idx in enumerate(self._reorder(block_indices)):
            # need to get the block size
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({device_function.sympy_expr(total_numel)})"
            if i + 1 < len(block_indices):
                expr = f"({expr}) % ({device_function.sympy_expr(numel)})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({device_function.sympy_expr(total_numel)})"
                )
            )
        return block_size_var, offsets_var, total_numel, statements

    def codegen_grid(self, state: CodegenState) -> None:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        dtype = CompileEnvironment.current().triton_index_type()
        state.add_statement(
            f"{offsets_var} = tl.program_id(0) * ({block_size_var}) + tl.arange(0, {block_size_var}).to({dtype})"
        )
        state.codegen.statements_stack[-1].extend(statements)
        state.device_function.set_grid_expr(
            expr_from_string(
                f"(triton.cdiv({HostFunction.current().sympy_expr(total_numel)}, {block_size_var}), 1, 1)"
            )
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        dtype = CompileEnvironment.current().triton_index_type()
        lid = self.new_var("lid")
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                f"range(tl.cdiv({state.device_function.sympy_expr(total_numel)}, {block_size_var}))"
            ),
            body=(
                body := [
                    statement_from_string(
                        f"{offsets_var} = {lid} * {block_size_var} + tl.arange(0, {block_size_var}).to({dtype})"
                    ),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
        )

    @classmethod
    def update_allow_flattened(
        cls, specs: list[BlockSizeSpec], shape: Sequence[sympy.Expr]
    ) -> None:
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = cls.get_block_index(x)
            if block_idx is not None:
                if block_idx in used_indices:
                    # multiple usages of the same block size??? bail out
                    for spec in specs:
                        spec.allow_flattened = False
                    return
                used_indices[block_idx] = i
        env = CompileEnvironment.current()
        for spec_idx, group in itertools.groupby(
            [
                bs
                for bs in env.block_sizes
                if isinstance(bs.block_size_source, LoopSpecBlockSizeSource)
            ],
            key=lambda x: x.block_size_source.loop_spec,
        ):
            spec = specs[spec_idx]
            if not spec.allow_flattened:
                continue
            block_indices = [bs.block_size_idx for bs in group]
            if len(block_indices) == 1 or not (
                all(x in used_indices for x in block_indices)
                or all(x not in used_indices for x in block_indices)
            ):
                # A shape must use all or none of the block indices in the group
                spec.allow_flattened = False
                continue
            for i, j in itertools.pairwise(block_indices):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    spec.allow_flattened = False
                    break

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            if (
                len(shape.block_indices) != 1
                or shape.block_indices[0] not in self.block_indices
            ):
                output.append(shape)
                continue
            assert shape.block_indices[0] == self.block_indices[0]
            for expected in self.block_indices[1:]:
                new_shape = shape_queue.popleft()
                assert len(new_shape.block_indices) == 1
                assert new_shape.block_indices[0] == expected
                shape = shape.combine(new_shape)
            output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_indices, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_indices, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def codegen_grid(self, state: CodegenState) -> None:
        block_indices = self.block_indices
        env = CompileEnvironment.current()
        device_function = state.device_function
        dtype = env.triton_index_type()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_indices)
        pids = self.select_pid_strategy()
        for i, (block_idx, block_size) in enumerate(
            reversed(self._reorder([*zip(block_indices, block_sizes, strict=True)]))
        ):
            numel = env.block_sizes[block_idx].numel
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)
            if block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                # TODO(jansel): need to check for conflict with user variable names since block_size_var is on host
                if state.device_function.constexpr_arg(block_size_var):
                    state.codegen.host_statements.append(
                        statement_from_string(
                            f"{block_size_var} = {HostFunction.current().literal_expr(block_size)}"
                        )
                    )
                state.add_statement(f"{offset_var} = {pid_var} * {block_size_var}")
                state.add_statement(
                    f"{index_var} = {offset_var} + tl.arange(0, ({block_size_var})).to({dtype})"
                )
            else:
                block_size_var = "1"
                dtype = CompileEnvironment.current().triton_index_type()
                state.add_statement(f"{offset_var} = {pid_var}")
                state.add_statement(
                    f"{index_var} = {offset_var} + tl.zeros([1], {dtype})"
                )
            mask_statement = self._setup_mask(  # pyre-ignore[16]
                state, block_idx, block_size, index_var, numel
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pids.append(ProgramID(pid_var, block_size_var, numel))
        pids.codegen(state)

    def select_pid_strategy(self) -> ProgramIDs:
        if 1 < len(self.block_indices) <= 3 and self.fn.config.use_yz_grid:
            return GridProgramIDs()
        return VirtualProgramIDs()

    def _to_ast(self, x: object) -> ast.AST:
        if isinstance(x, ast.AST):
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_indices = self.block_indices
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_indices)
        _, begins, ends, _ = state.ast_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        for block_idx, block_size, begin, end in self._reorder(
            [*zip(block_indices, block_sizes, begins, ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                if state.device_function.constexpr_arg(block_size_var):
                    state.codegen.host_statements.append(
                        statement_from_string(
                            f"{block_size_var} = {HostFunction.current().literal_expr(block_size)}"
                        )
                    )
            else:
                block_size_var = "1"
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    f"range(begin, end, {block_size_var})",
                    begin=self._to_ast(begin),
                    end=self._to_ast(end),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            extra_body = [
                statement_from_string(
                    f"{index_var} = {offset_var} + tl.arange(0, ({block_size_var})).to({dtype})"
                ),
            ]
            mask_statement = self._setup_mask(  # pyre-ignore[16]
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_indices, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
    ) -> ast.stmt | None:
        if (
            CompileEnvironment.current()
            .block_sizes[block_idx]
            .known_multiple(block_size)
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )
        return statement_from_string(
            f"{mask_var} = ({index_var}) < end", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(group_size=self.l2_grouping)
        return super().select_pid_strategy()


class NDGridTileStrategy(_BaseNDTileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_indices: list[int],
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_indices=block_indices,
            block_size=[1] * len(block_indices),  # pyre-ignore[6]
            loop_order=loop_order,
        )

    def mask_var(self, block_idx: int) -> str | None:
        return None

    def _setup_mask(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        return None


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_indices: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_indices=[*self.block_indices, *other.block_indices],
        )
