from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
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
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .host_function import HostFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import XYZProgramIDs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


@dataclasses.dataclass
class LoopDimInfo:
    end_var_name: str | None
    end_expr: sympy.Expr | None

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        hint = CompileEnvironment.current().shape_env.size_hint
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return hint(expected) == hint(self.end_expr)


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids


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
    block_ids: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"offset_{block_idx}", dce=True)
            for block_idx in block_ids
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

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            config.range_warp_specializes, block_idx, None
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if config.indexing == "tensor_descriptor":
            # Tensor descriptor + multi-stage pipelines in addition to unrolling tend to cause
            # CUDA "misaligned address" or "unspecified launch failure" errors.
            if range_num_stages > 0:
                range_num_stages = 0
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            # Unrolling can cause CUDA IMA with pipelining
            # We want to ensure new step size + pipeline is within bounds
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            step = range_unroll_factor * block_size
            last_offset = ((loop_numel - 1) // block_size) * block_size
            remainder = loop_numel - last_offset
            range_num_stages = min(
                max(1, int(math.ceil(remainder / step))), range_num_stages
            )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            kwargs.append(f"data_partition_factor={dpf_value}")

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()
        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self, state: CodegenState, use_proxy_ends: bool = False
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds."""
        env = CompileEnvironment.current()
        block_id_to_info = {}

        if use_proxy_ends:
            _, _, proxy_ends, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for block_idx, end in zip(self.block_ids, proxy_ends, strict=True):
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    end_var_name=None, end_expr=end_expr
                )
        else:
            for block_id in self.block_ids:
                end_expr = env.block_sizes[block_id].numel
                end_var_name = state.sympy_expr(end_expr)
                block_id_to_info[block_id] = LoopDimInfo(
                    end_var_name=end_var_name, end_expr=end_expr
                )

        return block_id_to_info

    def _setup_block_size_constexpr(
        self, state: CodegenState, block_size_var: str, block_size: SymIntLike
    ) -> None:
        """Helper to setup constexpr block size variable on host."""
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := env.block_sizes[block_id].from_config(
                    state.config
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        if env.known_multiple(
            functools.reduce(
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        ):
            self._mask_var: str | None = None
        else:
            self._mask_var = self.new_var("mask", dce=True)

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def _codegen_common(
        self, state: CodegenState
    ) -> tuple[str, str, sympy.Expr, list[ast.AST]]:
        offsets_var = self.new_var("offsets", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        statements = []

        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )
        return block_size_var, offsets_var, total_numel, statements

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        state.add_statement(
            f"{offsets_var} = tl.program_id(0) * ({block_size_var}) + tl.arange(0, {block_size_var}).to({dtype})"
        )
        state.codegen.statements_stack[-1].extend(statements)

        class TmpPid(ProgramIDs):
            def codegen_grid(self) -> ast.AST:
                return expr_from_string(
                    f"(triton.cdiv({HostFunction.current().sympy_expr(total_numel)}, {block_size_var}), 1, 1)"
                )

            def codegen(self, state: CodegenState) -> None:
                pass  # No-op implementation for TmpPid

            def total_pids_expr(self, *, is_device: bool) -> str:
                return "1"  # Simple implementation for TmpPid

        state.device_function.set_pid(TmpPid())

        block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(self, block_id_to_info=block_id_to_info)

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state
        )
        dtype = CompileEnvironment.current().triton_index_type()
        lid = self.new_var("lid")
        end_var = f"tl.cdiv({state.sympy_expr(total_numel)}, {block_size_var})"
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
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
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)

        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            if not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            ):
                flatten_loops.disable_block_id(block_ids[0])
                continue
            for i, j in itertools.pairwise(block_ids):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    flatten_loops.disable_block_id(block_ids[0])
                    break

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            if len(shape.block_ids) != 1 or shape.block_ids[0] not in self.block_ids:
                output.append(shape)
                continue
            assert shape.block_ids[0] == self.block_ids[0]
            for expected in self.block_ids[1:]:
                new_shape = shape_queue.popleft()
                assert len(new_shape.block_ids) == 1
                assert new_shape.block_ids[0] == expected
                shape = shape.combine(new_shape)
            output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
        else:
            begins = state.proxy_args[0]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)

        for i, (block_idx, block_size, begin) in enumerate(
            reversed(self._reorder([*zip(block_ids, block_sizes, begins, strict=True)]))
        ):
            numel = env.block_sizes[block_idx].numel
            device_function = state.device_function
            dtype = env.triton_index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
                state.add_statement(
                    f"{index_var} = ({offset_var} + tl.arange(0, ({block_size_var}))).to({dtype})"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
                state.add_statement(
                    f"{index_var} = {offset_var} + tl.zeros([1], {dtype})"
                )
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, numel
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(self, block_id_to_info=block_id_to_info)

    def select_pid_strategy(self) -> ProgramIDs:
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        if pid_type == "persistent_blocked":
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                return expr_from_string(f"{{value}}.to({to_dtype})", value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.triton_index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        _, begins, ends, _ = state.ast_args
        _, _, proxy_ends, _ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        for block_idx, block_size, begin, end, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=block_size_var,
                    ),
                    begin=self._to_ast(begin, to_dtype=dtype),
                    end=self._to_ast(end, to_dtype=dtype),
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
            mask_statement = self._setup_mask(  # pyright: ignore[reportAttributeAccessIssue]
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            body[:] = [*extra_body, *body]  # pyright: ignore[reportArgumentType,reportCallIssue]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
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
            f"{mask_var} = ({index_var}) < {{end}}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
