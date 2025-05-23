from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import threading
import typing
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol

import sympy
import torch
from torch._inductor.codegen.wrapper import pexpr

from .. import exc
from . import ast_extension
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .output_header import SOURCE_MODULE
from .source_location import SourceLocation
from .source_location import UnknownLocation
from .type_printer import print_ast
from .variable_origin import AttributeOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import NameOrigin
from .variable_origin import Origin

if TYPE_CHECKING:
    import types

    from .type_propagation import TypeInfo

    class _TLS(Protocol):
        functions: list[HostFunction]


tls: _TLS = typing.cast("_TLS", threading.local())


class GlobalImport(NamedTuple):
    value: object
    module: str
    alias: str
    member: str | None = None

    def __repr__(self) -> str:
        return f"<GlobalImport '{self.codegen()}'>"

    def codegen(self) -> str:
        if self.member is not None:
            if self.alias is not None:
                return f"from {self.module} import {self.member} as {self.alias}"
            return f"from {self.module} import {self.member}"
        if self.alias is not None:
            return f"import {self.module} as {self.alias}"
        return f"import {self.module}"


class SymbolOrigin(NamedTuple):
    origin: Origin
    fake_value: torch.Tensor | None = None

    def depth(self) -> int:
        return self.origin.depth()


class HostFunction:
    def __init__(
        self,
        fn: types.FunctionType,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> None:
        super().__init__()
        env = CompileEnvironment.current()
        self.fn = fn
        self.constexpr_args = constexpr_args
        self.location: SourceLocation = UnknownLocation()
        self.local_types: dict[str, TypeInfo] | None = None
        self.expr_to_origin: dict[sympy.Expr, SymbolOrigin] = {}
        self.tensor_to_origin: dict[torch.Tensor, Origin] = {}
        self.global_imports: dict[str, GlobalImport] = {}
        with self:
            source_indented = inspect.getsource(fn)
            source = textwrap.dedent(source_indented)
            self.column_offset: int = source_indented.index(source[0])
            root = ast.parse(source)
            assert isinstance(root, ast.Module)
            (root,) = root.body
            root = ast_extension.convert(root)
            assert isinstance(root, ast.FunctionDef)
            assert isinstance(root, ast_extension.ExtendedAST)
            self.location = root._location
            self.name: str = root.name
            self.args: ast.arguments = root.args
            self.body: list[ast.stmt] = root.body

            HostFunction.validate_ast(root)

            from .device_ir import lower_to_device_ir
            from .type_propagation import propagate_types

            propagate_types(self, fake_args)
            env.errors.raise_if_errors()
            env.finalize_config_spec()
            self.device_ir = lower_to_device_ir(self)

    @staticmethod
    def validate_ast(root: ast.FunctionDef) -> None:
        # There must always be at least one decorator otherwise we would not have gotten this far
        if len(root.decorator_list) > 1:
            # Decorators are allowed before the helion kernel decorator
            # but are not allowed after
            def get_decorator_name(decorator: ast.expr) -> str:
                if isinstance(decorator, ast.Name):
                    return decorator.id
                if isinstance(decorator, ast.Attribute):
                    return get_decorator_name(decorator.value)
                if isinstance(decorator, ast.Call):
                    return get_decorator_name(decorator.func)
                raise AssertionError(f"Unknown decorator: {decorator}")

            for idx, decorator in enumerate(root.decorator_list):
                # TODO(oulgen): this can break if someone did `import helion as helion2`
                if get_decorator_name(decorator) == "helion":
                    if idx != len(root.decorator_list) - 1:
                        raise exc.DecoratorAfterHelionKernelDecorator

    def global_scope_origin(self, name: str) -> AttributeOrigin:
        if SOURCE_MODULE not in self.global_imports:
            module_name = self.fn.__globals__["__name__"]
            module = sys.modules[module_name]
            assert module.__dict__ is self.fn.__globals__
            self.global_imports[SOURCE_MODULE] = GlobalImport(
                value=module,
                module=module_name,
                alias=SOURCE_MODULE,
            )
        return AttributeOrigin(GlobalOrigin(SOURCE_MODULE), name)

    def import_from_module(
        self, module_scope: dict[str, object], name: str
    ) -> AttributeOrigin:
        if module_scope is self.fn.__globals__:
            return self.global_scope_origin(name)
        module_name = module_scope["__name__"]
        assert isinstance(module_name, str)
        if module_name not in self.global_imports:
            module = sys.modules[module_name]
            assert module.__dict__ is module_scope
            alias = f"_global_source{len(self.global_imports)}"
            self.global_imports[module_name] = GlobalImport(
                value=module,
                module=module_name,
                alias=alias,
            )
        return AttributeOrigin(
            GlobalOrigin(self.global_imports[module_name].alias), name
        )

    def register_fake(self, obj: object, origin: Origin) -> object:
        value = CompileEnvironment.current().to_fake(obj, origin)
        if isinstance(value, torch.Tensor):
            self.tensor_to_origin[value] = origin
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            self.expr_to_origin[value._sympy_()] = SymbolOrigin(origin)
        return value

    def __repr__(self) -> str:
        return f"<HostFunction {self.name}>"

    def set_local_types(self, local_types: dict[str, TypeInfo]) -> None:
        fn = HostFunction.current()
        self.local_types = local_types
        for name, type_info in local_types.items():
            type_info.populate_symbol_origins(NameOrigin(name, fn))

    def sympy_expr(self, expr: sympy.Expr) -> str:
        expr = CompileEnvironment.current().shape_env.simplify(expr)
        if expr in self.expr_to_origin:
            return self.expr_to_origin[expr].origin.host_str()
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            origin = self.expr_to_origin[sym].origin
            replacements[sym] = sympy.Symbol(origin.host_str(), integer=True)
        return pexpr(expr.xreplace(replacements))

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        if isinstance(expr, list):
            return "[" + ", ".join(self.literal_expr(x) for x in expr) + "]"
        if isinstance(expr, tuple):
            return "(" + ", ".join(self.literal_expr(x) for x in expr) + ", )"
        return repr(expr)

    def debug_str(self) -> str:
        result = [
            print_ast(
                self.location.to_ast(
                    ast.FunctionDef(self.name, self.args, self.body, [], None)
                )
            ),
            self.device_ir.debug_str(),
        ]
        if error_str := CompileEnvironment.current().errors.report(strip_paths=True):
            result.extend(error_str)
        return "\n\n".join(result)

    def codegen_function_def(self, statements: list[ast.AST]) -> ast.FunctionDef:
        return ast_extension.create(
            ast.FunctionDef,
            name=self.name,
            args=self.args,
            body=statements,
            decorator_list=[],
            type_comment=None,
            returns=None,
            type_params=None,
        )

    def codegen_imports(self) -> list[ast.stmt]:
        return [
            statement_from_string(line.codegen())
            for line in self.global_imports.values()
        ]

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]
        self.location.__enter__()

    def __exit__(self, *args: object) -> None:
        self.location.__exit__(*args)
        tls.functions.pop()

    @staticmethod
    def current() -> HostFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None


class NoCurrentFunction(RuntimeError):
    pass
