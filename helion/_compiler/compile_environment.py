from __future__ import annotations

import collections
import dataclasses
import threading
import types
import typing
from typing import TYPE_CHECKING
from typing import Protocol

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import triton_type
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import exc
from ..language.constexpr import ConstExpr
from .error_reporting import ErrorReporting
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


class CompileEnvironment:
    """
    Global state for the duration of a compilation.
    There is a 1:1 mapping between this an BoundKernel,
    and a single CompileEnvironment will be used for multiple Configs.
    No config or codegen specific state should be stored here.
    """

    def __init__(self, device: torch.device, settings: Settings) -> None:
        from ..autotuner.config_spec import ConfigSpec

        super().__init__()
        self.device = device
        self.settings = settings
        self.errors = ErrorReporting(settings)
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

    def add_kernel_tensor_size(self, sizes: Sequence[int | torch.SymInt]) -> None:
        from .tile_strategy import TileStrategy

        for size in sizes:
            if isinstance(size, torch.SymInt):
                block_idx = TileStrategy.get_block_index(size)
                if block_idx is None:
                    value = self.shape_env.replace(size._sympy_())
                    if value.free_symbols:
                        raise exc.ShapeSpecializingAllocation
        self.kernel_tensor_sizes[(*map(_to_sympy, sizes),)] += 1

    def finalize_config_spec(self) -> None:
        from .tile_strategy import FlattenedTileStrategy

        for shape in self.kernel_tensor_sizes:
            FlattenedTileStrategy.update_allow_flattened(
                self.config_spec.block_size_specs, shape
            )

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
                block_size_idx=idx,
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
        for rdim in self.block_sizes:
            if rdim.reduction and rdim.size == size:
                return rdim
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
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint()
            # self.shape_env.guards.append(
            #     ShapeGuard(
            #         sympy.Ne(sym._sympy_(), 0),
            #         SLoc("create_block_var", current_location().format()),
            #         True,
            #     )
            # )
            # TODO(jansel): I was hoping the above would work, seems like some decomps require concrete values
            #               to determine zeroness.  Figure out a better way to do this.
            # pyre-ignore[29]
            self.shape_env.var_to_val[sym._sympy_()] = sympy.Integer(hint)
        assert isinstance(sym._sympy_(), sympy.Symbol)
        self.debug_shape_renames[sym._sympy_()] = sympy.Symbol(debug_name, integer=True)
        return sym

    def to_fake(self, obj: object, origin: Origin) -> object:
        if isinstance(obj, torch.Tensor):
            return self._to_fake_tensor(obj, origin.to_source())
        if isinstance(obj, (bool, int, float)):
            if isinstance(obj, bool):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    return self.shape_env.create_unbacked_symbool()
            if isinstance(obj, int):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    sym = self.shape_env.create_unbacked_symint()
                    # TODO(jansel): this is a hack to get us past some == 1 checks
                    #               we should probably have a better way to handle this
                    self.shape_env.var_to_val[sym._sympy_()] = sympy.sympify(8192)
                    return sym
            if isinstance(obj, float):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    return self.shape_env.create_unbacked_symfloat()
        if isinstance(
            obj,
            (torch.dtype, torch.device, types.BuiltinFunctionType, types.ModuleType),
        ):
            return obj
        if isinstance(obj, types.FunctionType):
            from .lift_closures import lift_closures

            return lift_closures(obj, origin)
        if isinstance(obj, ConstExpr):
            return obj.value
        if isinstance(obj, list):
            return [self.to_fake(e, origin) for e in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(
                **{  # pyre-ignore[6]
                    k: self.to_fake(e, origin)
                    for k, e in obj._asdict().items()  # pyre-ignore[16]
                }
            )
        if isinstance(obj, tuple):
            return tuple(self.to_fake(e, origin) for e in obj)
        if isinstance(obj, dict):
            return {k: self.to_fake(e, origin) for k, e in obj.items()}
        if dataclasses.is_dataclass(obj):
            return dataclasses.replace(
                obj,
                **{
                    k: self.to_fake(getattr(obj, k), origin)
                    for k in obj.__dataclass_fields__  # pyre-ignore[16]
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
            # pyre-ignore[6]
            return int(self.shape_env.size_hint(n._sympy_()))
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
        self.errors = ErrorReporting(self.settings)  # clear prior errors
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        tls.env = None
        self.fake_mode.__exit__(exc_type, exc_value, traceback)
        self.errors.raise_if_errors()

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

    block_size_idx: int
    size: torch.SymInt | int | AutoSize | None
    var: torch.SymInt
    reduction: bool
    block_size_source: BlockSizeSource

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

    def mark_alternate_size(self, size: torch.SymInt | int | None) -> None:
        """If a block size is used with a different size, we need to clear the hint to enable masking."""
        if isinstance(self.size, AutoSize):
            # The block size was created by hl.register_block_size, and we didn't know the size yet.
            self.size = size
            if isinstance(size, (int, torch.SymInt)) and isinstance(
                source := self.block_size_source, LoopSpecBlockSizeSource
            ):
                # update the size hint now that we know the size
                env = CompileEnvironment.current()
                env.config_spec.block_size_specs[source.loop_spec].update_hint(
                    source.dim, env.size_hint(size)
                )
        elif size is None or self.size is None or self.size != size:
            self.size = None

    def symbol(self) -> sympy.Symbol:
        return self.var._sympy_()

    def from_config(self, config: Config) -> int | torch.SymInt | None:
        return self.block_size_source.from_config(config)

    def from_config_assert(self, config: Config) -> int | torch.SymInt:
        val = self.from_config(config)
        assert val is not None
        return val

    def is_flattened(self, config: Config) -> bool:
        return self.block_size_source.is_flattened(config)

    def is_grid(self) -> bool:
        return self.block_size_source.is_grid()

    def get_order(self, config: Config, count: int) -> list[int]:
        return self.block_size_source.get_order(config, count)

    def l2_grouping(self, config: Config) -> int:
        return self.block_size_source.l2_grouping(config)

    def update_min_block(self, value: int, *, allow_flattened: bool = True) -> None:
        return self.block_size_source.update_min_block(
            value, allow_flattened=allow_flattened
        )


class BlockSizeSource:
    def from_config(self, config: Config) -> int | torch.SymInt | None:
        raise NotImplementedError

    def is_flattened(self, config: Config) -> bool:
        return False

    def is_grid(self) -> bool:
        return False

    def get_order(self, config: Config, count: int) -> list[int]:
        return [*range(count)]

    def l2_grouping(self, config: Config) -> int:
        return 1

    def update_min_block(self, value: int, *, allow_flattened: bool = True) -> None:
        return None


@dataclasses.dataclass
class FixedBlockSizeSource(BlockSizeSource):
    value: int | torch.SymInt

    def from_config(self, config: Config) -> int | torch.SymInt:
        return self.value


@dataclasses.dataclass
class GridBlockSizeSource(BlockSizeSource):
    def from_config(self, config: Config) -> int:
        raise NotImplementedError

    def is_grid(self) -> bool:
        return True


@dataclasses.dataclass
class LoopSpecBlockSizeSource(BlockSizeSource):
    loop_spec: int
    dim: int

    def from_config(self, config: Config) -> int:
        value = config.block_sizes[self.loop_spec]
        if isinstance(value, int):
            assert self.dim == 0
            return value
        return value[self.dim]

    def is_flattened(self, config: Config) -> bool:
        return isinstance(config.block_sizes[self.loop_spec], int)

    def get_order(self, config: Config, count: int) -> list[int]:
        env = CompileEnvironment.current()
        spec = env.config_spec.block_size_specs[self.loop_spec]
        if not spec.allow_reorder:
            return super().get_order(config, count)
        assert len(spec) == count
        order_offset = sum(
            [
                int(s.allow_reorder)
                for s in env.config_spec.block_size_specs[: self.loop_spec]
            ]
        )
        order = config.loop_orders[order_offset]
        assert len(order) == count
        return order

    def l2_grouping(self, config: Config) -> int:
        spec = CompileEnvironment.current().config_spec.block_size_specs[self.loop_spec]
        if spec.allow_l2_grouping:
            return config.l2_grouping
        return 1

    def update_min_block(self, value: int, *, allow_flattened: bool = True) -> None:
        """
        Update the minimum block size for the given block index, only increases the minimum size.
        """
        spec = CompileEnvironment.current().config_spec.block_size_specs[self.loop_spec]
        spec.update_min(self.dim, value)
        spec.allow_flattened = spec.allow_flattened and allow_flattened


@dataclasses.dataclass
class ReductionLoopBlockSizeSource(BlockSizeSource):
    reduction_loop: int

    def from_config(self, config: Config) -> int | None:
        return config.reduction_loops[self.reduction_loop]


def warning(warning: exc.BaseWarning | type[exc.BaseWarning]) -> None:
    CompileEnvironment.current().errors.add(warning)


def _to_sympy(x: int | torch.SymInt) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return x._sympy_()
    return sympy.sympify(x)
