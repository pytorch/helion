from __future__ import annotations

import ast
import contextlib
import dataclasses
import sys
import threading
import typing
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol

import sympy
import torch
from torch._inductor.codegen.wrapper import pexpr
from torch.utils._sympy.symbol import SymT
from torch.utils._sympy.symbol import symbol_is_type

from . import ast_extension
from .ast_extension import expr_from_string
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
    import inspect
    import types

    from .device_ir import DeviceIR
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


@dataclasses.dataclass
class KernelDefinition:
    """The kernel's structural definition.

    Holds the function, its AST, and parameter bindings. Populated by
    KernelCompiler.parse(). The AST body may be mutated by subsequent
    compilation passes (e.g. static loop unrolling).
    """

    fn: types.FunctionType
    constexpr_args: dict[str, object]
    name: str
    args: ast.arguments
    body: list[ast.stmt]
    params: inspect.BoundArguments


@dataclasses.dataclass
class CompilerState:
    """Mutable state accumulated during compilation passes.

    Tracks symbol and tensor provenance, import requirements, and
    resource allocation. Populated progressively during compilation,
    consumed by code generation.
    """

    expr_to_origin: dict[sympy.Expr, SymbolOrigin] = dataclasses.field(
        default_factory=dict
    )
    tensor_to_origin: dict[torch.Tensor, Origin] = dataclasses.field(
        default_factory=dict
    )
    global_imports: dict[str, GlobalImport] = dataclasses.field(default_factory=dict)
    rng_seed_slot_count: int = 0


class HostFunction:
    """Mutable compilation state for a @helion.kernel function.

    Composed of structured sub-states:

      - definition: KernelDefinition — function, AST, and parameter bindings
      - compiler_state: CompilerState — provenance tracking and imports
      - device_ir: DeviceIR — FX graphs from lowering

    Created and driven through the pipeline by KernelCompiler.
    Accessed by compiler passes via HostFunction.current().
    """

    def __init__(
        self,
        fn: types.FunctionType,
    ) -> None:
        super().__init__()
        # pyrefly: ignore [read-only]
        self._fn = fn
        self.location: SourceLocation = UnknownLocation()
        self.definition: KernelDefinition | None = None
        self.compiler_state: CompilerState = CompilerState()
        self._device_ir: DeviceIR | None = None
        # TODO(hinriksnaer): could be a local in KernelCompiler.parse()
        # if SourceLocation.from_ast() took code/column_offset explicitly.
        self._column_offset: int = 0

    # Backward-compatible accessors

    # TODO(hinriksnaer): migrate call sites to hf.definition.* and
    # hf.compiler_state.* directly, then remove these properties.

    @property
    def fn(self) -> types.FunctionType:
        if self.definition is not None:
            return self.definition.fn
        return self._fn

    @property
    def constexpr_args(self) -> dict[str, object]:
        assert self.definition is not None
        return self.definition.constexpr_args

    @property
    def name(self) -> str:
        assert self.definition is not None
        return self.definition.name

    @property
    def args(self) -> ast.arguments:
        assert self.definition is not None
        return self.definition.args

    @property
    def body(self) -> list[ast.stmt]:
        assert self.definition is not None
        return self.definition.body

    @body.setter
    def body(self, value: list[ast.stmt]) -> None:
        assert self.definition is not None
        self.definition.body = value

    @property
    def params(self) -> inspect.BoundArguments:
        assert self.definition is not None
        return self.definition.params

    @property
    def column_offset(self) -> int:
        return self._column_offset

    @property
    def device_ir(self) -> DeviceIR:
        assert self._device_ir is not None
        return self._device_ir

    @device_ir.setter
    def device_ir(self, value: DeviceIR) -> None:
        self._device_ir = value

    @property
    def expr_to_origin(self) -> dict[sympy.Expr, SymbolOrigin]:
        return self.compiler_state.expr_to_origin

    @property
    def tensor_to_origin(self) -> dict[torch.Tensor, Origin]:
        return self.compiler_state.tensor_to_origin

    @property
    def global_imports(self) -> dict[str, GlobalImport]:
        return self.compiler_state.global_imports

    @property
    def rng_seed_slot_count(self) -> int:
        return self.compiler_state.rng_seed_slot_count

    @rng_seed_slot_count.setter
    def rng_seed_slot_count(self, value: int) -> None:
        self.compiler_state.rng_seed_slot_count = value

    @staticmethod
    def _suppress_guards_if_profiler_enabled(
        env: CompileEnvironment,
    ) -> contextlib.AbstractContextManager[None]:
        if torch.autograd.profiler._is_profiler_enabled:
            return env.shape_env.suppress_guards()
        return contextlib.nullcontext()

    def global_scope_origin(self, name: str) -> AttributeOrigin:
        if SOURCE_MODULE not in self.global_imports:
            # pyrefly: ignore [missing-attribute]
            module_name = self.fn.__globals__["__name__"]
            module = sys.modules[module_name]
            # pyrefly: ignore [missing-attribute]
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
        # pyrefly: ignore [missing-attribute]
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

    def allocate_rng_seed_slot(self) -> int:
        seed_slot = self.rng_seed_slot_count
        self.rng_seed_slot_count += 1
        return seed_slot

    def set_local_types(self, local_types: dict[str, TypeInfo]) -> None:
        for name, type_info in local_types.items():
            type_info.populate_symbol_origins(NameOrigin(name, self))

    def sympy_expr(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        with contextlib.suppress(Exception):
            expr = env.shape_env.simplify(expr)
        expr = env.specialize_expr(expr)
        if not expr.free_symbols:
            return pexpr(expr)
        if expr in self.expr_to_origin:
            return self.expr_to_origin[expr].origin.host_str()
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            origin = self.expr_to_origin[sym].origin
            replacements[sym] = sympy.Symbol(origin.host_str(), integer=True)
        # pyrefly: ignore [bad-argument-type]
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
        ]
        if self._device_ir is not None:
            result.append(self._device_ir.debug_str())
        return "\n\n".join(result)

    def codegen_function_def(
        self,
        statements: list[ast.AST],
        extra_params: list[str] | None = None,
        removed_args: set[str] | None = None,
    ) -> ast.FunctionDef:
        # Rebuild defaults: Python aligns defaults to the *end* of args,
        # so removing an arg shifts alignment.
        if removed_args:
            old_args = self.args.args
            old_defaults = self.args.defaults
            n_no_default = len(old_args) - len(old_defaults)
            new_defaults = [
                old_defaults[i - n_no_default]
                for i, a in enumerate(old_args)
                if a.arg not in removed_args and i >= n_no_default
            ]
        else:
            new_defaults = self.args.defaults

        # Create a new arguments structure with _launcher kwarg-only parameter
        new_args = ast_extension.create(
            ast.arguments,
            posonlyargs=self.args.posonlyargs,
            args=[
                *(
                    a
                    for a in self.args.args
                    if not removed_args or a.arg not in removed_args
                ),
                *(
                    ast_extension.create(ast.arg, arg=n, annotation=None)
                    for n in (extra_params or [])
                ),
            ],
            vararg=self.args.vararg,
            kwonlyargs=[
                *self.args.kwonlyargs,
                ast_extension.create(
                    ast.arg,
                    arg="_launcher",
                    annotation=None,
                ),
            ],
            kw_defaults=[
                *self.args.kw_defaults,
                expr_from_string(
                    CompileEnvironment.current().backend.get_launcher_name()
                ),
            ],
            kwarg=self.args.kwarg,
            defaults=new_defaults,
        )

        return ast_extension.create(
            ast.FunctionDef,
            name=self.name,
            args=new_args,
            body=statements,
            decorator_list=[],
            type_comment=None,
            returns=None,
            type_params=None,
        )

    def codegen_call_function(self) -> ast.FunctionDef:
        def stringify(arg: object) -> str:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    return "()"
                parts = [stringify(a) for a in arg]
                return f"({','.join(parts)},)"
            if isinstance(arg, (int, bool, float)):
                return str(arg)
            if isinstance(arg, str):
                return f'"{arg}"'
            if isinstance(arg, torch.SymInt):
                return str(CompileEnvironment.current().size_hint(arg))
            if isinstance(arg, torch.SymFloat):
                if symbol_is_type(arg.node.expr, SymT.UNBACKED_FLOAT):
                    return "1.1"
                return str(arg.node._hint)
            if isinstance(arg, torch.SymBool):
                if not arg.node._hint:
                    return "False"
                return str(arg.node._hint)
            # TODO(oulgen): Support more types here
            return '"UNSUPPORTED TYPE - REPLACE"'

        inits = []
        for name, arg in self.params.arguments.items():
            if isinstance(arg, torch.Tensor):
                rhs = f"rand_strided(size={stringify(arg.size())}, stride={stringify(arg.stride())}, dtype={arg.dtype}, device='{arg.device}')"
            else:
                rhs = stringify(arg)
            inits.append(statement_from_string(f"{name} = {rhs}"))

        call_args = self.params.arguments.keys()
        statements = [
            statement_from_string("from torch._dynamo.testing import rand_strided"),
            *inits,
            statement_from_string(f"{self.name}({', '.join(call_args)})"),
        ]
        return ast_extension.create(
            ast.FunctionDef,
            name="call",
            args=[],
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
