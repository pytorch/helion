from __future__ import annotations

import ast
import collections
import contextlib
from typing import TYPE_CHECKING
from typing import NamedTuple

from torch.utils._ordered_set import OrderedSet

from .. import exc
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import dead_assignment_elimination
from .ast_read_writes import dead_expression_elimination
from .ast_read_writes import definitely_does_not_have_side_effects
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .helper_function import CodegenInterface
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .program_id import ForEachProgramID
from .tile_strategy import DeviceLoopState
from .variable_origin import ArgumentOrigin

if TYPE_CHECKING:
    from collections.abc import Iterator

    import sympy

    from ..runtime import Config
    from .host_function import HostFunction
    from .tile_strategy import DeviceLoopOrGridState
    from .type_propagation import TensorType


class GenerateAST(NodeVisitor, CodegenInterface):
    def __init__(self, func: HostFunction, config: Config) -> None:
        # Initialize NodeVisitor first
        NodeVisitor.__init__(self)

        # Initialize our attributes
        self.host_function = func
        self.host_statements: list[ast.AST] = []
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.active_device_loops: dict[int, list[DeviceLoopOrGridState]] = (
            collections.defaultdict(list)
        )
        self.next_else_block: list[ast.AST] | None = None

        # Now create device function and initialize CodegenInterface
        self.device_function = DeviceFunction(f"_helion_{func.name}", config, self)
        CodegenInterface.__init__(self, self.device_function)

    def offset_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        self.statements_stack[-1].append(stmt)

    def get_rng_seed_buffer_statements(self) -> list[ast.AST]:
        import_stmt = statement_from_string(
            "from torch._inductor import inductor_prims"
        )

        # Create host-side seed buffer with the required number of seeds
        seed_buffer_stmt = statement_from_string(
            f"_rng_seed_buffer = inductor_prims.seeds({self.device_function.rng_seed_count}, torch.accelerator.current_accelerator())"
        )

        return [import_stmt, seed_buffer_stmt]

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "v") -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr
        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            self.add_statement(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            return create(ast.Name, id=varname, ctx=ast.Load())

    def lift_symnode(
        self,
        expr: ast.AST,
        sym_expr: sympy.Expr,
        *,
        dce: bool = False,
        prefix: str = "symnode",
    ) -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr

        target_statements = self.statements_stack[-1]
        env = CompileEnvironment.current()
        # Identify every block dimension the symbolic value depends on so we know
        # which loop nests the expression depends on.
        dep_block_ids = {
            block_id
            for symbol in sym_expr.free_symbols
            if (block_id := env.get_block_id(symbol)) is not None
        }

        # Walk outward through the active device loops: as soon as we see a loop
        # whose block id appears in the dependency set we must stop, otherwise we
        # can safely hoist into that loop's outer prefix (which executes before the
        # loop body).
        for loop_state in reversed(self._active_loop_stack()):
            if dep_block_ids.intersection(loop_state.block_ids):
                break
            target_statements = loop_state.outer_prefix

        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            # Emit the temporary into the chosen statement list so the symbolic
            # expression is computed exactly once at the appropriate scope.
            target_statements.append(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            # Reuse the temporary everywhere else in the kernel body.
            return create(ast.Name, id=varname, ctx=ast.Load())

    def _active_loop_stack(self) -> list[DeviceLoopState]:
        seen: set[int] = set()
        stack: list[DeviceLoopState] = []
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, DeviceLoopState):
                    continue
                key = id(loop_state)
                if key not in seen:
                    stack.append(loop_state)
                    seen.add(key)
        return stack

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
        else:
            expr_to_var_info = self.device_function.expr_to_var_info
            # We don't want to reuse vars assigned in a nested scope, so copy it
            self.device_function.expr_to_var_info = expr_to_var_info.copy()
            self.statements_stack.append(new_statements)
            try:
                yield
            finally:
                self.statements_stack.pop()
                self.device_function.expr_to_var_info = expr_to_var_info

    @contextlib.contextmanager
    def set_on_device(self) -> Iterator[None]:
        assert self.on_device is False
        self.on_device = True
        prior = self.host_statements
        self.host_statements = self.statements_stack[-1]
        try:
            yield
        finally:
            self.on_device = False
            self.host_statements = prior

    @contextlib.contextmanager
    def add_device_loop(self, device_loop: DeviceLoopState) -> Iterator[None]:
        with self.set_statements(device_loop.inner_statements):
            for idx in device_loop.block_ids:
                active_loops = self.active_device_loops[idx]
                active_loops.append(device_loop)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            try:
                yield
            finally:
                for idx in device_loop.block_ids:
                    self.active_device_loops[idx].pop()
        self.statements_stack[-1].extend(device_loop.outer_prefix)
        self.add_statement(device_loop.for_node)
        self.statements_stack[-1].extend(device_loop.outer_suffix)

    def set_active_loops(self, device_grid: DeviceLoopOrGridState) -> None:
        for idx in device_grid.block_ids:
            self.active_device_loops[idx] = [device_grid]

    def generic_visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        fields = {}
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                fields[field] = new_list = []
                with self.set_statements(
                    new_list
                    if old_value and isinstance(old_value[0], ast.stmt)
                    else None
                ):
                    for item in old_value:
                        new_list.append(self.visit(item))  # mutation in visit
            elif isinstance(old_value, ast.AST):
                fields[field] = self.visit(old_value)
            else:
                fields[field] = old_value
        return node.new(fields)  # pyright: ignore[reportReturnType]

    def visit_For(self, node: ast.For) -> ast.AST | None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse

            if len(self.host_function.device_ir.root_ids) == 1:
                body = self.device_function.body
            else:
                assert len(self.host_function.device_ir.root_ids) > 1
                assert node._root_id is not None
                # Multiple top level for loops

                if node._root_id == 0:
                    self.device_function.set_pid(
                        ForEachProgramID(
                            self.device_function.new_var("pid_shared", dce=False)
                        )
                    )
                    self.device_function.body.extend(
                        self.device_function.pid.codegen_pid_init()  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
                    )
                if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                    body = []
                else:
                    # This is the last top level for, dont emit more if statements
                    assert self.next_else_block is not None
                    body = self.next_else_block
            with (
                self.set_on_device(),
                self.set_statements(body),
            ):
                iter_node = node.iter
                assert isinstance(iter_node, ExtendedAST)
                with iter_node:
                    assert isinstance(iter_node, ast.Call)
                    args = []
                    kwargs = {}
                    for arg_node in iter_node.args:
                        assert not isinstance(arg_node, ast.Starred)
                        assert isinstance(arg_node, ExtendedAST)
                        assert arg_node._type_info is not None
                        args.append(arg_node._type_info.proxy())
                    for kwarg_node in iter_node.keywords:
                        assert kwarg_node.arg is not None
                        assert isinstance(kwarg_node.value, ExtendedAST)
                        assert kwarg_node.value._type_info is not None
                        kwargs[kwarg_node.arg] = kwarg_node.value._type_info.proxy()
                    fn_node = iter_node.func
                    assert isinstance(fn_node, ExtendedAST)
                    assert fn_node._type_info is not None
                    fn = fn_node._type_info.proxy()
                    assert is_api_func(fn)
                    assert fn._codegen is not None
                    bound = fn._signature.bind(*args, **kwargs)
                    bound.apply_defaults()

                    from .inductor_lowering import CodegenState

                    state = CodegenState(
                        self,
                        fx_node=None,
                        proxy_args=[*bound.arguments.values()],
                        ast_args=None,  # pyright: ignore[reportArgumentType]
                    )

                    fn._codegen(state)
                assert node._root_id is not None
                codegen_call_with_graph(
                    self,
                    self.host_function.device_ir.get_root(
                        self.device_function.config,
                        self.host_function.device_ir.root_ids[node._root_id],
                    ),
                    [],
                )

                # Flush deferred RDIM definitions now that block sizes are determined
                # This ensures block size and rdim vars are defined in the correct order
                self.device_function.flush_deferred_rdim_defs(self)

                # If we are in a multi top level loop, for all loops except for the last one
                # emit ifthenelse blocks
                if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                    block = (
                        self.device_function.body
                        if self.next_else_block is None
                        else self.next_else_block
                    )
                    self.next_else_block = []
                    block.append(
                        create(
                            ast.If,
                            test=self.device_function.pid.codegen_test(state),  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
                            body=body,
                            orelse=self.next_else_block,
                        )
                    )
            if node._root_id == len(self.host_function.device_ir.root_ids) - 1:
                if self.device_function.pid is not None:
                    persistent_body = self.device_function.pid.setup_persistent_kernel(
                        self.device_function
                    )
                    if persistent_body is not None:
                        self.device_function.body = persistent_body  # pyright: ignore[reportAttributeAccessIssue]
                self.device_function.dead_code_elimination()
                return self.device_function.codegen_function_call()
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        if isinstance(node.ctx, ast.Load) and node._type_info is not None:
            origin = node._type_info.origin
            if (
                isinstance(origin, ArgumentOrigin)
                and origin.name in self.host_function.constexpr_args
            ):
                return expr_from_string(
                    repr(self.host_function.constexpr_args[origin.name])
                )
            if origin.needs_rename():
                # `x` => `_source_module.x`
                return expr_from_string(origin.host_str())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        from .type_propagation import CallableType
        from .type_propagation import SequenceType
        from .type_propagation import TileIndexType

        func_node = node.func
        assert isinstance(func_node, ExtendedAST)

        assert isinstance(node, ExtendedAST)
        env = CompileEnvironment.current()
        if self.on_device:
            pass
        elif isinstance(type_info := node._type_info, TileIndexType):
            block_info = env.block_sizes[type_info.block_id]
            return expr_from_string(
                self.host_function.literal_expr(
                    block_info.from_config(self.device_function.config)
                )
            )
        elif isinstance(type_info, SequenceType) and all(
            isinstance(x, TileIndexType) for x in type_info.unpack()
        ):
            values = type_info.unpack()
            block_infos = [env.block_sizes[x.block_id] for x in values]  # pyright: ignore[reportAttributeAccessIssue]
            return expr_from_string(
                self.host_function.literal_expr(
                    [x.from_config(self.device_function.config) for x in block_infos]
                )
            )
        elif (
            isinstance(fn_type_info := func_node._type_info, CallableType)
            and is_api_func(api := fn_type_info.value)
            and api._codegen is not None
        ):
            ast_args = []
            ast_kwargs = {}
            proxy_args = []
            proxy_kwargs = {}
            for arg in node.args:
                assert not isinstance(arg, ast.Starred)
                assert isinstance(arg, ExtendedAST)
                assert arg._type_info is not None
                ast_args.append(arg)
                proxy_args.append(arg._type_info.proxy())
            for kwarg in node.keywords:
                assert kwarg.arg is not None
                assert isinstance(kwarg.value, ExtendedAST)
                assert kwarg.value._type_info is not None
                ast_kwargs[kwarg.arg] = kwarg.value
                proxy_kwargs[kwarg.arg] = kwarg.value._type_info.proxy()
            ast_params = api._signature.bind(*ast_args, **ast_kwargs)
            proxy_params = api._signature.bind(*proxy_args, **proxy_kwargs)
            ast_params.apply_defaults()
            proxy_params.apply_defaults()
            return api._codegen(  # pyright: ignore[reportReturnType]
                CodegenState(
                    self,
                    None,
                    proxy_args=[*proxy_params.arguments.values()],
                    ast_args=[*ast_params.arguments.values()],
                )
            )
        return self.generic_visit(node)

    def host_dead_code_elimination(self) -> None:
        dce_vars: OrderedSet[str] = OrderedSet()
        for stmt in self.host_statements:
            if (
                isinstance(stmt, ast.Assign)
                and definitely_does_not_have_side_effects(stmt.value)
                and all(isinstance(name, ast.Name) for name in stmt.targets)
            ):
                for name in stmt.targets:
                    assert isinstance(name, ast.Name)
                    dce_vars.add(name.id)

        dead_assignment_elimination(self.host_statements, list(dce_vars))
        dead_expression_elimination(self.host_statements)


class TensorReference(NamedTuple):
    node: ast.AST
    name: str
    type_info: TensorType

    @property
    def is_host(self) -> bool:
        return self.type_info.origin.is_host()


class SubscriptIndexing(NamedTuple):
    tensor_ref: TensorReference
    index_expr: ast.AST
    mask_expr: ast.AST

    def has_mask(self) -> bool:
        return not (
            isinstance(self.mask_expr, ast.Constant) and self.mask_expr.value is None
        )


def emit_main_def() -> ast.stmt:
    return statement_from_string("""
if __name__ == "__main__":
    call()
    """)


def generate_ast(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.AST:
    with func:
        codegen = GenerateAST(func, config)
        with codegen.device_function:
            for stmt in func.body:
                codegen.add_statement(codegen.visit(stmt))
            kernel_def = codegen.device_function.codegen_function_def()
            codegen.host_dead_code_elimination()

            # Inject RNG seed buffer creation if needed
            rng_statements = (
                codegen.get_rng_seed_buffer_statements()
                if codegen.device_function.has_rng_ops()
                else []
            )
            final_host_statements = rng_statements + codegen.host_statements

            host_def = func.codegen_function_def(final_host_statements)

            call_def = []
            main_def = []
            if emit_repro_caller:
                call_def = [func.codegen_call_function()]
                main_def = [emit_main_def()]

            result = ast.Module(
                [
                    *func.codegen_imports(),
                    *codegen.device_function.codegen_helper_functions(),
                    *kernel_def,
                    host_def,
                    *call_def,
                    *main_def,
                ],
                [],
            )
            # break circular reference for better GC
            del codegen.device_function.codegen
            return result
