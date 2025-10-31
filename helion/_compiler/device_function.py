from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
import itertools
import math
import threading
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.codegen.triton import TritonPrinter
from torch.fx.graph import _Namespace

from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import ReadWrites
from .ast_read_writes import ast_rename
from .ast_read_writes import dead_assignment_elimination
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .output_header import reserved_names
from .variable_origin import BlockSizeOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from ..runtime.config import IndexingLiteral
    from .device_ir import HelperFunctionGraphInfo
    from .generate_ast import GenerateAST
    from .indexing_strategy import IndexingStrategy
    from .program_id import ProgramIDs

    _P = TypeVar("_P", bound="TensorPropertyArg")

    class _TLS(Protocol):
        functions: list[DeviceFunction]


tls: _TLS = cast("_TLS", threading.local())


class VarInfo(NamedTuple):
    """Information about a variable derived from a sympy expression."""

    name: str
    fx_node: torch.fx.Node


def find_block_size_symbols(
    expr: sympy.Expr,
) -> tuple[dict[sympy.Symbol, int], set[sympy.Symbol]]:
    """
    Find block size symbols in a sympy expression.

    Returns:
        tuple of (block_size_mapping, non_block_size_symbols) where:
        - block_size_mapping: dict mapping block size symbols to their block_id
        - non_block_size_symbols: set of symbols that are NOT block sizes
    """
    if not isinstance(expr, sympy.Expr):
        return {}, set()

    hf = HostFunction.current()
    block_sizes = {}
    non_block_size_symbols = set()

    for symbol in expr.free_symbols:
        origin_info = hf.expr_to_origin.get(symbol)  # pyright: ignore[reportArgumentType]
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            non_block_size_symbols.add(symbol)
        else:
            block_sizes[symbol] = origin_info.origin.block_id

    return block_sizes, non_block_size_symbols


def contains_only_block_size_symbols(expr: sympy.Expr) -> bool:
    """Check if expression contains only block size symbols (no other variables)."""
    _, non_block = find_block_size_symbols(expr)
    return len(non_block) == 0


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
    _host_str: str | None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError("TensorArg has no host representation")
        return self._host_str


@dataclasses.dataclass
class TensorDescriptorArg(TensorArg):
    # Permutation applied to make stride==1 dimension last
    permutation: list[int] | None = None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError(
                "TensorDescriptorArg is device-only and has no host representation"
            )
        return self._host_str

    @property
    def inverse_permutation(self) -> list[int]:
        """Get the inverse permutation to undo the applied permutation."""
        if (permutation := self.permutation) is None:
            raise RuntimeError("TensorDescriptorArg.permutation is None")
        inverse_perm = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_perm[p] = i
        return inverse_perm


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
    def __init__(self, name: str, config: Config, codegen: GenerateAST) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.codegen = codegen
        self.arguments: list[Argument] = []
        self.preamble: list[ast.AST] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._tensor_descriptor_args: dict[
            tuple[torch.Tensor, str], TensorDescriptorArg
        ] = {}
        self._expr_args: dict[sympy.Expr, SymbolArgument] = {}
        self._constexpr_args: dict[str, ConstExprArg] = {}
        self._constexpr_host_defs: set[str] = set()
        self._tensor_properties: dict[
            tuple[type[TensorPropertyArg], torch.Tensor, int], TensorPropertyArg
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        self.pid: ProgramIDs | None = None
        self.namespace: _Namespace = _Namespace()
        self.namespace._used_names.update(reserved_names())
        self.namespace._used_names.update(
            # used by triton run() method
            [
                "grid",
                "warmup",
                "num_warps",
                "num_stages",
            ]
            + [
                x.removeprefix("_triton_config_")
                for x in config
                if x.startswith("_triton_config_")
            ]
        )
        self._variable_renames: dict[str, list[str]] = {}
        self.dce_vars: list[str] = []
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.expr_to_var_info: dict[sympy.Expr, VarInfo] = {}
        self.deferred_rdim_defs: list[tuple[str, sympy.Expr]] = []

        from .helper_function import HelperFunctionManager

        self.helper_manager = HelperFunctionManager()

        from .tile_dispatch import TileStrategyDispatch

        self.tile_strategy: TileStrategyDispatch = TileStrategyDispatch(self, config)

        # Store indexing config to lazily create strategies per load/store
        self._indexing_config = config.indexing
        self.indexing_strategies: list[IndexingStrategy] = []

        self.rng_seed_count = 0
        self.device_load_index = 0
        self.device_store_index = 0
        # Single counter for both loads and stores for indexing assignment
        self.device_memory_op_index = 0
        self.rng_seed_buffer_param_name = None

    def get_indexing_strategy(self, index: int) -> IndexingStrategy:
        from typing import cast

        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        # Expand strategies list if needed
        while len(self.indexing_strategies) <= index:
            idx = len(self.indexing_strategies)

            if isinstance(self._indexing_config, str):
                # Single string: all loads/stores use the same strategy
                if not self.indexing_strategies:
                    strategy = IndexingStrategy.select(
                        cast("IndexingLiteral", self._indexing_config)
                    )
                else:
                    strategy = self.indexing_strategies[0]
            elif isinstance(self._indexing_config, list) and self._indexing_config:
                # List: one strategy per load/store
                assert idx < len(self._indexing_config), (
                    f"Load/Store operation {idx} exceeds indexing config length "
                    f"{len(self._indexing_config)}. Please specify indexing for all loads and stores."
                )
                strategy = IndexingStrategy.select(
                    cast("IndexingLiteral", self._indexing_config[idx])
                )
            else:
                # Empty/default: use pointer
                strategy = PointerIndexingStrategy()

            self.indexing_strategies.append(strategy)

        return self.indexing_strategies[index]

    def has_rng_ops(self) -> bool:
        """Check if this kernel uses any RNG operations."""
        return self.rng_seed_count > 0 and self.rng_seed_buffer_param_name is not None

    def allocate_rng_seed(self) -> int:
        """Allocate a new RNG seed index and ensure buffer argument exists.

        Returns:
            The seed index for this RNG operation.
        """
        seed_index = self.rng_seed_count
        self.rng_seed_count += 1

        # Ensure seed buffer parameter name exists
        if self.rng_seed_buffer_param_name is None:
            self.rng_seed_buffer_param_name = self.new_var("rng_seed_buffer")

        return seed_index

    def block_size_var(self, block_id: int) -> str | None:
        key = (block_id,)

        # Block size var could be used outside of a hl.tile loop, and at that point
        # no tile strategy has populated the cache yet, so we must lazily create
        # the constexpr argument here and lift it as device function argument;
        # later strategies will reuse the cached name or intentionally replace it
        # (e.g. flattened loops, reductions).
        if key not in self.block_size_var_cache:
            env = CompileEnvironment.current()
            block_value = env.block_sizes[block_id].from_config(self.config)

            if block_value is None:
                return None

            var_name = self.new_var(f"_BLOCK_SIZE_{block_id}")
            self.block_size_var_cache[key] = var_name
            self.constexpr_arg_with_host_def(var_name, block_value)

        return self.block_size_var_cache[key]

    def try_map_block_symbols_to_vars(self, expr: sympy.Expr) -> sympy.Expr | None:
        """Try to map all block size symbols in expression to their variable names.

        Returns:
            - The expression with symbols replaced if ALL symbols are block sizes and have variables
            - None if the expression contains non-block symbols or unmapped block symbols
        """
        block_mapping, non_block_symbols = find_block_size_symbols(expr)

        # Can't map if there are non-block symbols
        if non_block_symbols:
            return None

        # No symbols to map - return as-is
        if not block_mapping:
            return expr

        # Try to map all block symbols to their variables
        var_map = {}
        for symbol, block_id in block_mapping.items():
            block_var = self.block_size_var(block_id)
            if not block_var:
                # Can't map this block symbol - fail
                return None
            var_map[symbol] = sympy.Symbol(block_var, integer=True)

        # Successfully mapped all symbols
        return expr.xreplace(var_map)

    def merge_variable_names(self, a: str, b: str) -> None:
        name_group = [
            *self._variable_renames.get(a, [a]),
            *self._variable_renames.get(b, [b]),
        ]
        for n in name_group:
            self._variable_renames[n] = name_group

    def set_pid(self, pid: ProgramIDs) -> None:
        assert self.pid is None, "pid already set"
        self.pid = pid

    def sympy_expr(self, expr: sympy.Expr) -> str:
        expr = CompileEnvironment.current().shape_env.simplify(expr)
        if not expr.free_symbols:
            return texpr(expr)
        if expr in self.expr_to_var_info:
            return self.expr_to_var_info[expr].name
        expr_to_origin = HostFunction.current().expr_to_origin
        if expr in expr_to_origin:
            return self._lift_sympy_arg(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):  # pyright: ignore[reportAttributeAccessIssue]
            assert isinstance(sym, sympy.Symbol)
            if sym in self.expr_to_var_info:
                replacements[sym] = sympy.Symbol(
                    self.expr_to_var_info[sym].name, integer=True
                )
            else:
                assert sym in expr_to_origin, f"no origin found for {sym.name}"
                replacements[sym] = sympy.Symbol(
                    self._lift_sympy_arg(sym), integer=True
                )
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
            result = self.block_size_var(origin.origin.block_id)
            assert result is not None
            return result
        if isinstance(origin.origin, GridOrigin):
            return self.codegen.offset_var(origin.origin.block_id)
        return self.expr_arg(expr, origin.origin).name

    def user_sympy_expr(self, expr: sympy.Expr) -> str:
        """A sympy expression that flows into user computations."""
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):  # pyright: ignore[reportAttributeAccessIssue]
            assert isinstance(sym, sympy.Symbol)
            block_idx = CompileEnvironment.current().get_block_id(sym)
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
    ) -> TensorDescriptorArg:
        host_function = HostFunction.current()
        block_size_expr = ", ".join(map(self.literal_expr, block_size))
        key = (fake_value, block_size_expr)
        if key not in self._tensor_descriptor_args:
            origin = host_function.tensor_to_origin[fake_value]
            desc_name = self.new_var(origin.suggest_var_name() + "_desc")
            env = CompileEnvironment.current()

            # Find which dimension has stride==1
            stride_one_dim = [*map(env.size_hint, fake_value.stride())].index(1)

            # Determine if we need permutation (stride==1 dimension is not last)
            permutation = None
            if stride_one_dim != fake_value.ndim - 1:
                # Create permutation to move stride==1 dimension to last position
                permutation = [*range(fake_value.ndim)]
                permutation.pop(stride_one_dim)
                permutation.append(stride_one_dim)

            # Create the regular tensor arg and size/stride args
            tensor_arg = self.tensor_arg(fake_value)
            size_args = [
                self.tensor_size(fake_value, i) for i in range(fake_value.ndim)
            ]
            stride_args = [
                self.tensor_stride(fake_value, i) for i in range(fake_value.ndim)
            ]

            # Apply permutation if needed
            if permutation is not None:
                size_args = [size_args[i] for i in permutation]
                stride_args = [stride_args[i] for i in permutation]
                block_size = [block_size[i] for i in permutation]
                # Update block_size_expr for the permuted order
                block_size_expr = ", ".join(map(self.literal_expr, block_size))

            # Add tl.make_tensor_descriptor call to preamble
            sizes = ", ".join([arg.name for arg in size_args])
            strides = ", ".join([arg.name for arg in stride_args])

            tensor_descriptor_fn_name = get_tensor_descriptor_fn_name()
            descriptor_stmt = statement_from_string(
                f"{desc_name} = {tensor_descriptor_fn_name}({tensor_arg.name}, [{sizes}], [{strides}], [{block_size_expr}])"
            )
            self.preamble.append(descriptor_stmt)

            arg = TensorDescriptorArg(
                desc_name,
                fake_value,
                None,  # No host_str since this is device-only
                permutation,
            )
            # Don't add to self.arguments since this is device-only
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

    def constexpr_arg(self, name: str, value: object | None = None) -> bool:
        """Create a constexpr argument, returns True if created, False if already exists."""
        if name in self._constexpr_args:
            return False
        host_str = name if value is None else self._format_constexpr_value(value)
        self._constexpr_args[name] = rv = ConstExprArg(name, host_str)
        self.arguments.append(rv)
        return True

    def constexpr_arg_with_host_def(self, name: str, value: object) -> None:
        """Create a constexpr argument and add its host-side definition if needed."""
        created = self.constexpr_arg(name, value)
        host_expr = self._constexpr_args[name].host_str()
        if created or name not in self._constexpr_host_defs:
            self.codegen.host_statements.append(
                statement_from_string(f"{name} = {host_expr}")
            )
        self._constexpr_host_defs.add(name)

    def _format_constexpr_value(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return repr(value)

        # Extract sympy expression from torch symbolic types
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()

        # Handle sympy expressions (sanitize by replacing triton_helpers functions)
        if isinstance(value, sympy.Expr):
            sanitized = value.replace(  # pyright: ignore[reportAttributeAccessIssue]
                lambda node: isinstance(node, sympy.Function)
                and getattr(node.func, "__name__", "")
                == "triton_helpers.div_floor_integer",
                lambda node: sympy.floor(node.args[0] / node.args[1]),  # pyright: ignore[reportAttributeAccessIssue]
            ).replace(  # pyright: ignore[reportAttributeAccessIssue]
                lambda node: isinstance(node, sympy.Function)
                and getattr(node.func, "__name__", "")
                == "triton_helpers.remainder_integer",
                lambda node: sympy.Mod(node.args[0], node.args[1]),  # pyright: ignore[reportAttributeAccessIssue]
            )
            expr = cast("sympy.Expr", sanitized)
            return HostFunction.current().sympy_expr(expr)

        return HostFunction.current().literal_expr(value)

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

    def codegen_function_def(self) -> list[ast.stmt]:
        prefix = []
        if self._tensor_descriptor_args:
            prefix.append(
                statement_from_string("helion.runtime.set_triton_allocator()")
            )

        args = [arg.arg_def_node() for arg in self.sorted_args()]
        if self.has_rng_ops():
            # Add the seed buffer as a pointer parameter to kernel signature
            assert self.rng_seed_buffer_param_name is not None
            args.append(create_arg(self.rng_seed_buffer_param_name))

        return [
            *prefix,
            ast_rename(
                create(
                    ast.FunctionDef,
                    name=self.name,
                    args=create_arguments(args),
                    body=[*self.preamble, *self.body],
                    decorator_list=[expr_from_string("triton.jit")],
                    type_params=[],
                ),
                {k: v[0] for k, v in self._variable_renames.items()},
            ),
        ]

    def codegen_function_call(self) -> ast.AST:
        args = []
        for arg in self.sorted_args():
            if isinstance(arg, ConstExprArg) and arg.name in self._constexpr_host_defs:
                args.append(arg.name)
            else:
                args.append(arg.host_str())

        if self.has_rng_ops():
            # Pass the host-side seed buffer variable to the kernel
            args.append("_rng_seed_buffer")

        # Workaround for triton bug: warp_specialize requires at least 4 warps
        # See: https://github.com/triton-lang/triton/issues/7354
        num_warps = self.config.num_warps
        if any(self.config.range_warp_specializes):
            num_warps = max(4, num_warps)

        args.extend(
            [
                f"num_warps={num_warps}",
                f"num_stages={self.config.num_stages}",
            ]
            + [
                f"{x.removeprefix('_triton_config_')}={self.config[x]}"
                for x in self.config
                if x.startswith("_triton_config_")
            ]
        )
        pid = self.pid
        assert pid is not None
        # TODO(jansel): we should run CSE this statement
        call_statement = statement_from_string(
            f"_launcher({self.name}, {{call_grid_expr}}, {', '.join(args)})",
            call_grid_expr=pid.codegen_grid(),
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
            rw = ReadWrites.from_list([*self.preamble, *self.body])
            dead_assignment_elimination(self.body, self.dce_vars, 1, rw)
            dead_assignment_elimination(self.preamble, self.dce_vars, 1, rw)

        # drop any unused args
        args_to_remove = {
            arg.name
            for arg in self.arguments
            if arg.name not in rw.reads  # pyright: ignore[reportPossiblyUnboundVariable]
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

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> None:
        """Register a helper function to be generated at global scope."""
        name = self.namespace.create_name(helper_graph_info.name, None)
        self.helper_manager.register_helper_function(helper_graph_info, name)

    def codegen_helper_functions(self) -> list[ast.stmt]:
        """Generate helper function definitions at global scope."""
        return self.helper_manager.codegen_helper_functions()

    def flush_deferred_rdim_defs(self, codegen: GenerateAST) -> None:
        """Add all deferred RDIM definitions to host statements."""
        for var_name, expr in self.deferred_rdim_defs:
            stmt = statement_from_string(
                f"{var_name} = triton.next_power_of_2({HostFunction.current().sympy_expr(expr)})"
            )
            codegen.host_statements.append(stmt)
        self.deferred_rdim_defs.clear()

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


class HelionTritonPrinter(TritonPrinter):
    """Custom Triton printer that avoids wrapping float literals in tl.full().

    Inductor's default TritonPrinter prints SymPy Float as a 0-D Triton value
    via tl.full([], <val>, tl.float64). We override this to emit the raw numeric
    literal, letting downstream type promotion and casts handle dtype.
    """

    def _print_Float(self, expr: sympy.Expr) -> str:
        return str(expr)

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        return f"{expr} + 0.0"


def texpr(expr: sympy.Expr) -> str:
    return HelionTritonPrinter().doprint(expr)
