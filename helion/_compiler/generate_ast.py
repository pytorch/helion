from __future__ import annotations

import ast
import collections
import contextlib
from typing import TYPE_CHECKING
from typing import NamedTuple

from .. import exc
from ..language._decorators import is_api_func
from ..runtime.precompile_shim import make_precompiler
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .variable_origin import ArgumentOrigin

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..runtime import Config
    from .host_function import HostFunction
    from .tile_strategy import DeviceLoopOrGridState
    from .tile_strategy import DeviceLoopState
    from .type_propagation import TensorType


class GenerateAST(NodeVisitor):
    def __init__(self, func: HostFunction, config: Config) -> None:
        super().__init__()
        self.host_function = func
        self.host_statements: list[ast.AST] = []
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.device_function = DeviceFunction(f"_{func.name}_kernel", config)
        self.active_device_loops: dict[int, list[DeviceLoopOrGridState]] = (
            collections.defaultdict(list)
        )

    def offset_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

    def add_statement(self, stmt: ast.AST | str) -> None:
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        self.statements_stack[-1].append(stmt)

    def tmpvar(self, dce: bool = False) -> str:
        return self.device_function.unique_name("v", dce=dce)

    def lift(self, expr: ast.AST, dce: bool = False) -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr
        with expr:
            varname = self.tmpvar(dce=dce)
            self.add_statement(statement_from_string(f"{varname} = expr", expr=expr))
            return create(ast.Name, id=varname, ctx=ast.Load())

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
        else:
            self.statements_stack.append(new_statements)
            try:
                yield
            finally:
                self.statements_stack.pop()

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
            for idx in device_loop.block_indices:
                active_loops = self.active_device_loops[idx]
                active_loops.append(device_loop)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            try:
                yield
            finally:
                for idx in device_loop.block_indices:
                    self.active_device_loops[idx].pop()
        self.statements_stack[-1].extend(device_loop.outer_prefix)
        self.add_statement(device_loop.for_node)
        self.statements_stack[-1].extend(device_loop.outer_suffix)

    def set_active_loops(self, device_grid: DeviceLoopOrGridState) -> None:
        for idx in device_grid.block_indices:
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
        return node.new(fields)

    def visit_For(self, node: ast.For) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse
            with (
                self.set_on_device(),
                self.set_statements(self.device_function.body),
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
                        args.append(arg_node._type_info.proxy())
                    for kwarg_node in iter_node.keywords:
                        assert kwarg_node.arg is not None
                        assert isinstance(kwarg_node.value, ExtendedAST)
                        kwargs[kwarg_node.arg] = kwarg_node.value._type_info.proxy()
                    fn_node = iter_node.func
                    assert isinstance(fn_node, ExtendedAST)
                    fn = fn_node._type_info.proxy()
                    assert is_api_func(fn)
                    assert fn._codegen is not None
                    bound = fn._signature.bind(*args, **kwargs)
                    bound.apply_defaults()

                    from .inductor_lowering import CodegenState

                    fn._codegen(
                        CodegenState(
                            self,
                            fx_node=None,
                            proxy_args=[*bound.arguments.values()],
                            ast_args=None,
                        ),
                    )
                codegen_call_with_graph(
                    self,
                    self.host_function.device_ir.get_root(self.device_function.config),
                    [],
                )
            self.device_function.dead_code_elimination()
            return self.device_function.codegen_function_call()
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
            block_info = env.block_sizes[type_info.block_size_idx]
            return expr_from_string(
                self.host_function.literal_expr(
                    block_info.from_config(self.device_function.config)
                )
            )
        elif isinstance(type_info, SequenceType):
            values = type_info.unpack()
            if all(isinstance(x, TileIndexType) for x in values):
                block_infos = [env.block_sizes[x.block_size_idx] for x in values]
                return expr_from_string(
                    self.host_function.literal_expr(
                        [
                            x.from_config(self.device_function.config)
                            for x in block_infos
                        ]
                    )
                )
        elif (
            isinstance(fn_type_info := func_node._type_info, CallableType)
            and is_api_func(api := fn_type_info.value)
            and api._codegen is not None
        ):
            proxy_args = []
            proxy_kwargs = {}
            for arg in node.args:
                assert not isinstance(arg, ast.Starred)
                assert isinstance(arg, ExtendedAST)
                assert arg._type_info is not None
                proxy_args.append(arg._type_info.proxy())
            for kwarg in node.keywords:
                assert kwarg.arg is not None
                assert isinstance(kwarg.value, ExtendedAST)
                assert kwarg.value._type_info is not None
                proxy_kwargs[kwarg.arg] = kwarg.value._type_info.proxy()
            proxy_params = api._signature.bind(*proxy_args, **proxy_kwargs)
            proxy_params.apply_defaults()
            return api._codegen(
                CodegenState(
                    self,
                    None,
                    proxy_args=[*proxy_params.arguments.values()],
                )
            )
        return self.generic_visit(node)


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


def codegen_precompile_def(
    host_def: ast.FunctionDef, device_function_name: str
) -> ast.FunctionDef:
    """
    Generate a precompile function definition for the given host function.
    The precompile function is the same as the normal function, but the call to the
    kernel is replaced with a call to make_precompiler.

    :param host_def: The host function definition to that is used to call the kernel.
    :param device_function_name: The name of the device function to be called.
    :return: A transformed function definition with the kernel call replaced.
    """

    def transform(node: ExtendedAST) -> ExtendedAST:
        nonlocal found_calls
        assert not node._is_kernel_call
        fields = node.fields()
        for key, value in [*fields.items()]:
            if isinstance(value, list):
                new_list = []
                for item in value:
                    assert isinstance(item, ExtendedAST)
                    if item._is_kernel_call:
                        with item:
                            found_calls += 1
                            new_list.append(
                                statement_from_string(
                                    f"from {make_precompiler.__module__} import make_precompiler"
                                )
                            )
                            assert isinstance(item, ast.Expr)
                            value = item.value
                            assert isinstance(value, ExtendedAST)
                            new_list.append(
                                create(
                                    ast.Return,
                                    value=value.copy(
                                        func=expr_from_string(
                                            f"make_precompiler({device_function_name})"
                                        )
                                    ),
                                )
                            )
                            break
                    new_list.append(transform(item))
                fields[key] = new_list
            elif isinstance(value, ExtendedAST):
                fields[key] = transform(value)
        return node.new(fields)

    found_calls = 0
    assert isinstance(host_def, ExtendedAST)
    new_fn = transform(host_def)
    assert isinstance(new_fn, ast.FunctionDef)
    new_fn.name = f"_{host_def.name}_make_precompiler"
    assert found_calls == 1
    return new_fn


def generate_ast(func: HostFunction, config: Config) -> ast.AST:
    with func:
        codegen = GenerateAST(func, config)
        with codegen.device_function:
            for stmt in func.body:
                codegen.add_statement(codegen.visit(stmt))
            CompileEnvironment.current().errors.raise_if_errors()
            kernel_def = codegen.device_function.codegen_function_def()
            host_def = func.codegen_function_def(codegen.host_statements)
            precompile_def = codegen_precompile_def(
                host_def, codegen.device_function.name
            )
            return ast.Module(
                [
                    *func.codegen_imports(),
                    kernel_def,
                    host_def,
                    precompile_def,
                ],
                [],
            )
