from __future__ import annotations

import collections
import contextlib
import dataclasses
import sys
import threading
import types
import typing
from typing import TYPE_CHECKING
from typing import Protocol

import sympy
import torch
from torch._dynamo.source import EphemeralSource
from torch._dynamo.source import LocalSource
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import triton_type
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import exc
from ..language.constexpr import ConstExpr
from .loop_dependency_checker import LoopDependencyChecker
from .source_location import SourceLocation
from .source_location import current_location
from .variable_origin import BlockSizeOrigin
from .variable_origin import Origin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing_extensions import Self

    from torch._guards import Source

    from .. import Config
    from ..runtime.settings import Settings

    class _TLS(Protocol):
        env: CompileEnvironment | None


tls: _TLS = typing.cast("_TLS", threading.local())


class HelionKernelSource(EphemeralSource):
    """Ephemeral source that formats as a kernel file location."""

    def __init__(self, location: SourceLocation) -> None:
        super().__init__(desc=None)
        self.location = location

    def name(self) -> str:  # type: ignore[override]
        formatted = self.location.format().rstrip("\n")
        if not formatted:
            return ""
        return "\nHelion kernel stack:\n" + formatted


def _current_symbol_source() -> EphemeralSource | None:
    location = current_location()
    if not location:
        return None
    return HelionKernelSource(location)


class CompileEnvironment:
    """
    Global state for the duration of a compilation.
    There is a 1:1 mapping between this and a BoundKernel,
    and a single CompileEnvironment will be used for multiple Configs.
    No config or codegen specific state should be stored here.
    """

    def __init__(self, device: torch.device, settings: Settings) -> None:
        from ..autotuner.config_spec import ConfigSpec

        super().__init__()
        self.device = device
        self.settings = settings
        self.shape_env = ShapeEnv(
            specialize_zero_one=True,
            duck_shape=False,
            assume_static_by_default=settings.static_shapes,
        )
        # TODO(jansel): check for guards in the shapeenv
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        self.input_sources: dict[torch.Tensor, Source] = {}
        self.block_sizes: list[BlockSizeInfo] = []
        self.debug_shape_renames: dict[sympy.Expr, sympy.Expr] = {}
        self.config_spec = ConfigSpec()
        self.kernel_tensor_sizes: dict[tuple[sympy.Expr, ...], int] = (
            collections.Counter()
        )
        self.specialized_vars: set[sympy.Symbol] = set()
        self.loop_dependency_checker = LoopDependencyChecker()
        self._symint_cache: dict[object, torch.SymInt] = {}
        self.device_load_count = (
            0  # Track number of loads in all device code for eviction policy tuning
        )

    def add_kernel_tensor_size(self, sizes: Sequence[int | torch.SymInt]) -> None:
        from .device_function import contains_only_block_size_symbols

        for size in sizes:
            if isinstance(size, torch.SymInt):
                block_idx = self.get_block_id(size)
                if block_idx is None:
                    value = self.shape_env.replace(size._sympy_())
                    if value.free_symbols and not contains_only_block_size_symbols(
                        value
                    ):
                        raise exc.ShapeSpecializingAllocation
        self.kernel_tensor_sizes[(*map(_to_sympy, sizes),)] += 1

    def finalize_config_spec(self) -> None:
        from .tile_strategy import FlattenedTileStrategy

        for shape in self.kernel_tensor_sizes:
            FlattenedTileStrategy.update_allow_flattened(shape)
        self._disable_range_num_stages_for_aliasing()
        self.config_spec._remove_duplicates()

    def _disable_range_num_stages_for_aliasing(self) -> None:
        """
        Disable range_num_stages choices if any kernel argument name is both read and written.

        Workaround for https://github.com/triton-lang/triton/issues/8259
        """

        if not self.config_spec.range_num_stages:
            return

        from .ast_read_writes import ReadWrites
        from .host_function import HostFunction

        host_fn = HostFunction.current()
        rw = ReadWrites.from_list(host_fn.body)
        if not (rw.reads and rw.writes):
            return

        arg_names = set(host_fn.params.arguments.keys())
        if set(rw.reads) & set(rw.writes) & arg_names:
            self.config_spec.range_num_stages.clear()

    def allocate_block_size(
        self,
        size: int | torch.SymInt | AutoSize | None,
        *,
        reduction: bool = False,
        source: BlockSizeSource,
        hint: int = 64,
    ) -> int:
        idx = len(self.block_sizes)
        self.block_sizes.append(
            info := BlockSizeInfo(
                block_id=idx,
                size=size,
                var=self.create_block_var(
                    f"block_size_{idx}" if not reduction else f"rdim_{idx}",
                    hint=hint,
                ),
                reduction=reduction,
                block_size_source=source,
            )
        )

        from .host_function import HostFunction
        from .host_function import SymbolOrigin

        HostFunction.current().expr_to_origin[info.symbol()] = SymbolOrigin(
            origin=BlockSizeOrigin(idx),
        )
        return idx

    def allocate_reduction_dimension(self, size: torch.SymInt | int) -> BlockSizeInfo:
        # Check if this size is already a registered block size
        if isinstance(size, torch.SymInt):
            from .host_function import HostFunction

            expr = size._sympy_()
            origin_info = HostFunction.current().expr_to_origin.get(expr)
            if origin_info and isinstance(origin_info.origin, BlockSizeOrigin):
                block_idx = origin_info.origin.block_id
                # Return the existing block size if it's a reduction dimension
                if self.block_sizes[block_idx].reduction:
                    return self.block_sizes[block_idx]

        # Check for existing reduction dimensions with the same size
        for rdim in self.block_sizes:
            if rdim.reduction and rdim.size == size:
                return rdim

        # Allocate a new reduction dimension
        rdim_idx = self.allocate_block_size(
            size,
            reduction=True,
            source=ReductionLoopBlockSizeSource(
                sum([int(bs.reduction) for bs in self.block_sizes])
            ),
            hint=next_power_of_2(self.size_hint(size)),
        )
        return self.block_sizes[rdim_idx]

    def create_block_var(self, debug_name: str, hint: int = 64) -> torch.SymInt:
        source = _current_symbol_source()
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint(source=source)
            # self.shape_env.guards.append(
            #     ShapeGuard(
            #         sympy.Ne(sym._sympy_(), 0),
            #         SLoc("create_block_var", current_location().format()),
            #         True,
            #     )
            # )
            # TODO(jansel): I was hoping the above would work, seems like some decomps require concrete values
            #               to determine zeroness.  Figure out a better way to do this.

            self.shape_env.var_to_val[sym._sympy_()] = sympy.Integer(hint)
        assert isinstance(sym._sympy_(), sympy.Symbol)
        self.debug_shape_renames[sym._sympy_()] = sympy.Symbol(debug_name, integer=True)
        return sym

    def create_unbacked_symint(self, hint: int = 8192) -> torch.SymInt:
        source = _current_symbol_source()
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint(source=source)
            # TODO(jansel): this is a hack to get us past some == 1 checks
            #               we should probably have a better way to handle this
            self.shape_env.var_to_val[sym._sympy_()] = sympy.sympify(hint)
            return sym

    def cached_create_unbacked_symint(
        self, key: Sequence[object], hint: int = 8192
    ) -> torch.SymInt:
        """Create an unbacked symint with caching based on a key.

        This ensures that the same key always returns the same unbacked
        symint, which is crucial to allow simplification of expressions
        for things like tile_begin.

        Args:
            key: The cache key (should be sequence of hashables and unique for the desired symint)
            hint: Hint value for the symint

        Returns:
            A consistent unbacked symint for the given key
        """

        key = tuple([x._sympy_() if hasattr(x, "_sympy_") else x for x in key])  # pyright: ignore[reportAttributeAccessIssue]
        result = self._symint_cache.get(key)
        if result is None:
            result = self.create_unbacked_symint(hint)
            self._symint_cache[key] = result
        return result

    def to_fake(self, obj: object, origin: Origin) -> object:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return self._to_fake_tensor(obj, origin.to_source())
        if isinstance(obj, (bool, int, float)):
            if isinstance(obj, bool):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    return self.shape_env.create_unbacked_symbool()
            if isinstance(obj, int):
                return self.create_unbacked_symint()
            if isinstance(obj, float):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    return self.shape_env.create_unbacked_symfloat()
        if isinstance(
            obj,
            (
                torch.dtype,
                torch.device,
                types.BuiltinFunctionType,
                types.ModuleType,
                type,
            ),
        ):
            return obj
        # Handle functions and Kernel objects
        from ..runtime.kernel import Kernel

        if isinstance(obj, (types.FunctionType, Kernel)):
            from .helper_function import extract_helper_function
            from .lift_closures import lift_closures

            fn = extract_helper_function(obj)
            return lift_closures(fn, origin)
        # Handle GraphModule - treat it like a function
        if isinstance(obj, torch.fx.GraphModule):
            # GraphModule can be treated like a callable function
            # We return it as-is since it will be called during execution
            return obj
        if isinstance(obj, ConstExpr):
            return obj.value
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            return [self.to_fake(e, origin) for e in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(
                **{
                    k: self.to_fake(e, origin)
                    for k, e in obj._asdict().items()  # pyright: ignore[reportAttributeAccessIssue]
                }
            )
        if isinstance(obj, tuple):
            return tuple(self.to_fake(e, origin) for e in obj)
        if isinstance(obj, dict):
            return {k: self.to_fake(e, origin) for k, e in obj.items()}
        if dataclasses.is_dataclass(obj):
            return dataclasses.replace(
                obj,  # pyright: ignore[reportArgumentType]
                **{
                    k: self.to_fake(getattr(obj, k), origin)
                    for k in obj.__dataclass_fields__
                },
            )

        raise TypeError(f"unsupported argument type {type(obj)} ({origin})")

    def _to_fake_tensor(self, tensor: torch.Tensor, source: Source) -> torch.Tensor:
        assert CompileEnvironment.current() is self
        assert not self.fake_mode.is_our_fake(tensor)
        if self.settings.static_shapes:
            result = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            result = self.fake_mode.fake_tensor_converter.from_real_tensor(
                self.fake_mode, tensor, shape_env=self.shape_env, source=source
            )
        self.input_sources[result] = source
        if isinstance(source, LocalSource):
            for i, s in enumerate(result.size()):
                if isinstance(s, torch.SymInt) and isinstance(
                    s._sympy_(), sympy.Symbol
                ):
                    self.debug_shape_renames[s._sympy_()] = sympy.Symbol(
                        f"{source.local_name}_size{i}", integer=True
                    )
        return result

    def size_hint(self, n: int | torch.SymInt) -> int:
        if isinstance(n, torch.SymInt):
            expr = n._sympy_()
            if _has_unbacked(expr):
                # If the size is a symbolic expression with unbacked symbols, then the shape environment
                # hint will be wrong since we assign a default value to unbacked symbols.  Return a default hint.
                return 8192

            return int(self.shape_env.size_hint(n._sympy_()))  # pyright: ignore[reportArgumentType]
        assert isinstance(n, int)
        return n

    def known_equal(self, a: int | torch.SymInt, b: int | torch.SymInt) -> bool:
        if isinstance(a, torch.SymInt) or isinstance(b, torch.SymInt):
            sa = a._sympy_() if isinstance(a, torch.SymInt) else a
            sb = b._sympy_() if isinstance(b, torch.SymInt) else b
            if sa == sb:
                return True
            res = self.shape_env._maybe_evaluate_static(sympy.Eq(sa, sb))
            if res is None:
                return False
            return bool(res)
        return a == b

    def known_multiple(self, a: sympy.Expr, b: int | torch.SymInt) -> bool:
        if isinstance(a, (int, sympy.Integer)) and isinstance(b, int):
            return (int(a) % b) == 0
        return False

    def triton_index_type(self) -> str:
        """tl.int32 or tl.int64 depending on Settings()"""
        return triton_type(self.settings.index_dtype)

    def sympy_debug(self, expr: sympy.Expr) -> str:
        return str(expr.xreplace(self.debug_shape_renames))

    def __enter__(self) -> Self:
        assert getattr(tls, "env", None) is None, "CompileEnvironment already active"
        self.fake_mode.__enter__()
        tls.env = self
        self.loop_dependency_checker = LoopDependencyChecker()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        tls.env = None
        self.fake_mode.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def current() -> CompileEnvironment:
        try:
            if (env := tls.env) is not None:
                return env
        except AttributeError:
            pass
        raise NoCurrentEnvironment from None

    @staticmethod
    def has_current() -> bool:
        try:
            CompileEnvironment.current()
            return True
        except NoCurrentEnvironment:
            return False

    def get_block_id(self, size: int | torch.SymInt | sympy.Basic) -> int | None:
        """
        Get the block ID associated with a given size expression.

        This method determines if a size expression corresponds to a registered block size
        in the current compilation environment. It looks up the origin information of
        symbolic expressions to find their associated block IDs.

        Args:
            size: The size expression to check. Can be an integer, torch.SymInt, or sympy.Basic.

        Returns:
            The block ID if the size corresponds to a registered block size, None otherwise.
        """
        if isinstance(size, torch.SymInt):
            return self.get_block_id(size._sympy_())
        if isinstance(size, sympy.Symbol):
            from .host_function import HostFunction

            origin_info = HostFunction.current().expr_to_origin.get(size)
            if origin_info is not None and isinstance(
                origin_info.origin,
                BlockSizeOrigin,
            ):
                return origin_info.origin.block_id
        return None

    def resolve_block_id(self, size: object) -> int | None:
        """Best-effort lookup of a block id for ``size``.

        Falls back to matching constant reduction dimensions if ``get_block_id``
        cannot resolve the identifier directly.
        """

        if isinstance(size, (int, torch.SymInt, sympy.Expr)):
            block_id = self.get_block_id(size)
            if block_id is not None:
                return block_id
        else:
            block_id = None

        if isinstance(size, torch.SymInt):
            expr: sympy.Expr | None = size._sympy_()
        elif isinstance(size, int):
            expr = sympy.Integer(size)
        elif isinstance(size, sympy.Expr):
            expr = sympy.simplify(size)
        else:
            expr = None

        if expr is None or getattr(expr, "free_symbols", None):
            return None

        for info in reversed(self.block_sizes):
            if info.reduction and info.size_matches(expr):
                return info.block_id
        return None


class NoCurrentEnvironment(RuntimeError):
    pass


class AutoSize:
    """A marker used to delay setting the size of a block until it is known."""


@dataclasses.dataclass
class BlockSizeInfo:
    """
    Information about a block size.
    Used to track the block size for a given dimension.
    """

    block_id: int
    size: torch.SymInt | int | AutoSize | None
    var: torch.SymInt
    reduction: bool
    block_size_source: BlockSizeSource
    debug_names: set[str] = dataclasses.field(default_factory=set)

    def add_debug_name(self, name: str) -> None:
        if not name:
            return
        self.debug_names.add(name)

    @property
    def numel(self) -> sympy.Expr:
        assert isinstance(self.size, (int, torch.SymInt))
        return _to_sympy(self.size)

    def known_multiple(self, block_size: int | torch.SymInt) -> bool:
        if block_size == 1:
            return True
        if not isinstance(self.size, (int, torch.SymInt)):
            return False
        return CompileEnvironment.current().known_multiple(self.numel, block_size)

    def size_hint(self) -> int:
        size = self.size
        assert isinstance(size, (int, torch.SymInt))
        return CompileEnvironment.current().size_hint(size)

    def size_matches(self, numel: sympy.Expr | None) -> bool:
        if numel is None or not isinstance(self.size, (int, torch.SymInt)):
            return False
        return numel == self.numel

    def mark_alternate_size(self, size: torch.SymInt | int | None) -> None:
        """If a block size is used with a different size, we need to clear the hint to enable masking."""
        if isinstance(self.size, AutoSize):
            # The block size was created by hl.register_block_size, and we didn't know the size yet.
            self.size = size
            if size is not None:
                env = CompileEnvironment.current()
                # Refresh the var_to_val hint to match the resolved block size
                hint = env.size_hint(size)
                env.shape_env.var_to_val[self.symbol()] = sympy.Integer(hint)
                with contextlib.suppress(KeyError):
                    # update the size hint now that we know the size
                    env.config_spec.block_sizes.block_id_lookup(
                        self.block_id
                    ).update_hint(hint)
        elif size is None or self.size is None or self.size != size:
            self.size = None

    def symbol(self) -> sympy.Symbol:
        return self.var._sympy_()

    def from_config(self, config: Config) -> int | torch.SymInt | None:
        return self.block_size_source.from_config(config, self)

    def from_config_assert(self, config: Config) -> int | torch.SymInt:
        val = self.from_config(config)
        assert val is not None
        return val

    def is_flattened(self, config: Config) -> bool:
        spec = CompileEnvironment.current().config_spec
        return spec.flatten_loops.config_get(config.flatten_loops, self.block_id, False)

    def update_min_block(self, value: int, *, allow_flattened: bool = True) -> None:
        spec = CompileEnvironment.current().config_spec
        if not allow_flattened:
            spec.flatten_loops.disable_block_id(self.block_id)
        with contextlib.suppress(KeyError):
            spec.block_sizes.block_id_lookup(self.block_id).update_min(value)


class BlockSizeSource:
    def from_config(
        self, config: Config, block_size_info: BlockSizeInfo
    ) -> int | torch.SymInt | None:
        raise NotImplementedError

    def l2_grouping(self, config: Config) -> int:
        return 1


@dataclasses.dataclass
class FixedBlockSizeSource(BlockSizeSource):
    value: int | torch.SymInt

    def from_config(
        self, config: Config, block_size_info: BlockSizeInfo
    ) -> int | torch.SymInt:
        return self.value


@dataclasses.dataclass
class LoopSpecBlockSizeSource(BlockSizeSource):
    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int:
        env = CompileEnvironment.current()
        size = block_size_info.size
        if isinstance(size, (int, torch.SymInt)) and env.known_equal(size, 1):
            return 1
        index = env.config_spec.block_sizes.block_id_to_index(block_size_info.block_id)
        return config.block_sizes[index]


@dataclasses.dataclass
class ReductionLoopBlockSizeSource(BlockSizeSource):
    reduction_loop: int

    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int | None:
        if (
            len(config.reduction_loops) <= self.reduction_loop
            or config.reduction_loops[self.reduction_loop] is None
        ):
            return next_power_of_2(block_size_info.size_hint())
        return config.reduction_loops[self.reduction_loop]


def warning(warning: exc.BaseWarning | type[exc.BaseWarning]) -> None:
    """Print a warning to stderr if it's not in the ignore list."""
    env = CompileEnvironment.current()
    if callable(warning):
        warning = warning()

    if not isinstance(warning, exc.BaseWarning):
        raise TypeError(f"expected BaseWarning, got {type(warning)}")

    # Check if this warning type should be ignored
    if not isinstance(warning, tuple(env.settings.ignore_warnings)):
        print(f"WARNING[{type(warning).__name__}]: {warning.args[0]}", file=sys.stderr)


def _to_sympy(x: int | torch.SymInt) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return x._sympy_()
    return sympy.sympify(x)


def _has_unbacked(expr: sympy.Expr) -> bool:
    return any(n.name.startswith("u") for n in expr.free_symbols)  # pyright: ignore[reportAttributeAccessIssue]


def format_shape(shape: tuple[object, ...]) -> str:
    def _format_dim(dim: object) -> str:
        if isinstance(dim, torch.SymInt):
            env = CompileEnvironment.current()
            block_id = env.get_block_id(dim)
            if block_id is not None and (
                names := sorted(env.block_sizes[block_id].debug_names)
            ):
                return f"{' or '.join(names)} (symbol: {dim})"
        return str(dim)

    return "(" + ", ".join(_format_dim(d) for d in shape) + ")"
