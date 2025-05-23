from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
import itertools
import math
import threading
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.codegen.triton import texpr
from torch.fx.graph import _Namespace

from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import ReadWrites
from .ast_read_writes import ast_delete_assignments
from .ast_read_writes import ast_rename
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .output_header import reserved_names
from .tile_strategy import TileStrategy
from .variable_origin import BlockSizeOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config

    _P = TypeVar("_P", bound="TensorPropertyArg")

    class _TLS(Protocol):
        functions: list[DeviceFunction]


tls: _TLS = cast("_TLS", threading.local())


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError

    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name)

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)],)


@dataclasses.dataclass
class TensorArg(Argument):
    fake_value: torch.Tensor
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


@dataclasses.dataclass
class TensorDescriptorArg(TensorArg):
    pass


@dataclasses.dataclass
class TensorPropertyArg(Argument):
    tensor_arg: TensorArg
    dim: int

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)], self.tensor_arg.name, self.dim)


class TensorSizeArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStrideArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


@dataclasses.dataclass
class NumericArgument(Argument):
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


class ConstExprArg(NumericArgument):
    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name, "tl.constexpr")


@dataclasses.dataclass
class SymbolArgument(NumericArgument):
    pass


class StaticShape(Argument):
    def __init__(self, val: int) -> None:
        super().__init__(repr(val))


_sort_order: dict[type[Argument], int] = {
    TensorDescriptorArg: 0,
    TensorArg: 0,
    TensorSizeArg: 1,
    TensorStrideArg: 2,
    SymbolArgument: 3,
    ConstExprArg: 4,
}


class DeviceFunction:
    def __init__(self, name: str, config: Config) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.arguments: list[Argument] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._tensor_descriptor_args: dict[
            tuple[torch.Tensor, str], TensorDescriptorArg
        ] = {}
        self._expr_args: dict[sympy.Expr, SymbolArgument] = {}
        self._constexpr_args: dict[str, ConstExprArg] = {}
        self._tensor_properties: dict[
            tuple[type[TensorPropertyArg], torch.Tensor, int], TensorPropertyArg
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        self.grid_expr: ast.AST | None = None
        self.namespace: _Namespace = _Namespace()
        self.namespace._used_names.update(reserved_names())
        self._variable_renames: dict[str, list[str]] = {}
        self.dce_vars: list[str] = []
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}

        from .indexing_strategy import IndexingStrategy
        from .tile_dispatch import TileStrategyDispatch

        self.tile_strategy: TileStrategyDispatch = TileStrategyDispatch(self, config)
        self.indexing_strategy: IndexingStrategy = IndexingStrategy.select(config)

    def block_size_var(self, block_size_idx: int) -> str | None:
        return self.block_size_var_cache.get((block_size_idx,))

    def merge_variable_names(self, a: str, b: str) -> None:
        name_group = [
            *self._variable_renames.get(a, [a]),
            *self._variable_renames.get(b, [b]),
        ]
        for n in name_group:
            self._variable_renames[n] = name_group

    def set_grid_expr(self, grid_expr: ast.AST) -> None:
        assert self.grid_expr is None, "grid_expr already set"
        self.grid_expr = grid_expr

    def sympy_expr(self, expr: sympy.Expr) -> str:
        expr_to_origin = HostFunction.current().expr_to_origin
        expr = CompileEnvironment.current().shape_env.simplify(expr)
        if not expr.free_symbols:
            return texpr(expr)
        if expr in expr_to_origin:
            return self._lift_sympy_arg(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            assert sym in expr_to_origin, f"no origin found for {sym.name}"
            replacements[sym] = sympy.Symbol(self._lift_sympy_arg(sym), integer=True)
        return texpr(expr.xreplace(replacements))

    def _lift_sympy_arg(self, expr: sympy.Expr) -> str:
        origin = HostFunction.current().expr_to_origin[expr]
        if isinstance(origin.origin, TensorSizeOrigin):
            assert origin.fake_value is not None
            arg = self.tensor_size(
                origin.fake_value,
                origin.origin.key,
            )
            return arg.name
        if isinstance(origin.origin, BlockSizeOrigin):
            result = self.block_size_var(origin.origin.block_size_idx)
            assert result is not None
            return result
        return self.expr_arg(expr, origin.origin).name

    def user_sympy_expr(self, expr: sympy.Expr) -> str:
        """A sympy expression that flows into user computations."""
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):
            assert isinstance(sym, sympy.Symbol)
            block_idx = TileStrategy.get_block_index(sym)
            if block_idx is not None:
                replacements[sym] = self.tile_strategy.user_size(block_idx)
        if replacements:
            expr = expr.xreplace(replacements)
        return self.sympy_expr(expr)

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        if isinstance(expr, float) and not math.isfinite(expr):
            return f"float('{expr}')"
        return repr(expr)

    def unique_name(self, prefix: str, dce: bool = False) -> str:
        return self.new_var(f"{prefix}_{next(self._unique_counter[prefix])}", dce=dce)

    def new_var(self, name: str, *, dce: bool = False) -> str:
        name = self.namespace.create_name(name, None)
        if dce:
            self.dce_vars.append(name)
        return name

    def tensor_arg(
        self, fake_value: torch.Tensor, prefer_name: str | None = None
    ) -> TensorArg:
        if fake_value not in self._tensor_args:
            origin = HostFunction.current().tensor_to_origin[fake_value]
            arg = TensorArg(
                self.new_var(prefer_name or origin.suggest_var_name()),
                fake_value,
                origin.host_str(),
            )
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    def tensor_descriptor_arg(
        self, fake_value: torch.Tensor, block_size: list[int | torch.SymInt]
    ) -> TensorArg:
        host_function = HostFunction.current()
        block_size_expr = ", ".join(
            map(HostFunction.current().literal_expr, block_size)
        )
        key = (fake_value, block_size_expr)
        if key not in self._tensor_descriptor_args:
            origin = host_function.tensor_to_origin[fake_value]
            arg = TensorDescriptorArg(
                self.new_var(origin.suggest_var_name() + "_desc"),
                fake_value,
                f"TensorDescriptor.from_tensor({origin.host_str()}, [{block_size_expr}])",
            )
            self.arguments.append(arg)
            self._tensor_descriptor_args[key] = arg
        return self._tensor_descriptor_args[key]

    def expr_arg(self, sym: sympy.Expr, origin: Origin) -> SymbolArgument:
        if sym not in self._expr_args:
            arg = SymbolArgument(
                name=self.new_var(origin.suggest_var_name()),
                _host_str=origin.host_str(),
            )
            self.arguments.append(arg)
            self._expr_args[sym] = arg
        return self._expr_args[sym]

    def constexpr_arg(self, name: str, host_str: str | None = None) -> bool:
        """Create a constexpr argument, returns True if created, False if already exists."""
        if name in self._constexpr_args:
            return False
        self._constexpr_args[name] = rv = ConstExprArg(name, host_str or name)
        self.arguments.append(rv)
        return True

    def _tensor_property(
        self,
        prop_cls: type[_P],
        fake_value: torch.Tensor,
        dim: int,
        prefix: str,
    ) -> _P:
        # TODO(jansel): dedupe based on sympy expressions
        key = (prop_cls, fake_value, dim)
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"{arg.name}_{prefix}_{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> Argument:
        if isinstance(v := fake_value.size(dim), int) or isinstance(
            v._sympy_(), sympy.Integer
        ):
            return StaticShape(int(v))
        return self._tensor_property(TensorSizeArg, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> Argument:
        if (
            isinstance(v := fake_value.stride(dim), int)
            and CompileEnvironment.current().settings.static_shapes
        ):
            return StaticShape(v)
        return self._tensor_property(TensorStrideArg, fake_value, dim, "stride")

    def sorted_args(self) -> list[Argument]:
        self.arguments.sort(key=lambda arg: arg.sort_key())
        return self.arguments

    def codegen_function_def(self) -> ast.FunctionDef:
        return ast_rename(
            create(
                ast.FunctionDef,
                name=self.name,
                args=create_arguments(
                    [arg.arg_def_node() for arg in self.sorted_args()]
                ),
                body=self.body,
                decorator_list=[expr_from_string("triton.jit")],
                type_params=[],
            ),
            {k: v[0] for k, v in self._variable_renames.items()},
        )

    def codegen_function_call(self) -> ast.AST:
        args = [arg.host_str() for arg in self.sorted_args()]
        args.extend(
            [
                f"num_warps={self.config.num_warps}",
                f"num_stages={self.config.num_stages}",
            ]
        )
        grid_expr = self.grid_expr
        assert grid_expr is not None
        # TODO(jansel): we should run CSE this statement
        call_statement = statement_from_string(
            f"{self.name}[__call_grid_expr]({', '.join(args)})",
            __call_grid_expr=grid_expr,
        )
        assert isinstance(call_statement, ExtendedAST)
        # Mark the kernel call we can find it in codegen_precompile_def
        call_statement._is_kernel_call = True
        return call_statement

    def dead_code_elimination(self) -> None:
        """
        Remove variables that are not used in the function body.
        """

        for _ in range(8):
            rw = ReadWrites.from_list(self.body)
            to_remove = set()
            for name in self.dce_vars:
                if name in rw.writes and name not in rw.reads:
                    to_remove.add(name)
            if not to_remove:
                break
            self.body[:] = ast_delete_assignments(self.body, to_remove)

        # drop any unused args
        args_to_remove = {
            arg.name for arg in self.arguments if arg.name not in rw.reads
        }
        if args_to_remove:
            self.arguments = [
                arg for arg in self.arguments if arg.name not in args_to_remove
            ]
            for cache in cast(
                "list[dict[object, Argument]]",
                [
                    self._tensor_args,
                    self._tensor_descriptor_args,
                    self._expr_args,
                    self._tensor_properties,
                ],
            ):
                for k, v in [*cache.items()]:
                    if v.name in args_to_remove:
                        del cache[k]

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]

    def __exit__(self, *args: object) -> None:
        tls.functions.pop()

    @staticmethod
    def current() -> DeviceFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None
