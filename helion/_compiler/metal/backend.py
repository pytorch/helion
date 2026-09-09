"""MetalBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

import torch

from ... import exc
from ..backend import Backend
from ..backend import _largest_divisor_at_most
from ..cute.backend import CuteBackend
from ._constants import MAX_THREAD_AXES
from ._constants import MAX_THREADS_PER_THREADGROUP

if TYPE_CHECKING:
    from torch._inductor.ops_handler import OpsHandler

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..device_function import Argument
    from ..device_function import DeviceFunction
    from ..tile_strategy import TileStrategy

    InductorOpOverrides = OpsHandler[Any]


class MetalBackend(Backend):
    """Metal Shading Language (MSL) code generation backend for macOS."""

    @staticmethod
    def _get_dtype_to_metal() -> dict[torch.dtype, str]:
        from torch._inductor.codegen.mps import DTYPE_TO_METAL

        return DTYPE_TO_METAL

    _ACC_TYPE: ClassVar[dict[torch.dtype, str]] = {
        torch.float16: "float",
        torch.bfloat16: "float",
        torch.float32: "float",
        torch.int8: "int",
        torch.int16: "int",
        torch.int32: "int",
        torch.int64: "long",
        torch.uint8: "uint",
        torch.bool: "int",
    }

    _SUPPORTED_CONFIG_KEYS: frozenset[str] = frozenset(
        {
            "block_sizes",
            "num_threads",
            "num_warps",
            "reduction_loops",
        }
    )

    @property
    def name(self) -> str:
        return "metal"

    def dtype_str(self, dtype: torch.dtype) -> str:
        dtype_map = self._get_dtype_to_metal()
        if dtype not in dtype_map:
            raise exc.BackendUnsupported(self.name, f"dtype: {dtype}")
        return dtype_map[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype not in self._ACC_TYPE:
            raise exc.BackendUnsupported(self.name, f"acc_type for: {dtype}")
        return self._ACC_TYPE[dtype]

    @property
    def function_decorator(self) -> str:
        return "metal_jit"

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_metal_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "_default_metal_launcher": (
                "from helion.runtime import default_metal_launcher"
                " as _default_metal_launcher"
            ),
            "metal_jit": ("from helion._compiler.metal.metal_jit import metal_jit"),
        }

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        return "uint"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        # The outer parens matter: Python parses ``static_cast<T>(x)`` as a
        # chained comparison, and without them ``a + static_cast<T>(x)`` binds
        # as ``(a + static_cast) < T > x``, which the MSL walker cannot
        # recognize as a cast.
        return f"(static_cast<{dtype_str}>({expr_str}))"

    def _tid(self, axis: int) -> str:
        """The ``tid`` component for a thread axis, rejecting a fourth one.

        Metal provides three.  A kernel wanting more -- a 3-D tile whose
        innermost dimension is also fully sliced, so that it claims an axis of
        its own -- would otherwise index past the end of the ``uint3``, which
        compiles cleanly and returns garbage.
        """
        if axis >= MAX_THREAD_AXES:
            raise exc.BackendUnsupported(
                self.name,
                f"thread axis {axis}: a Metal threadgroup has "
                f"{MAX_THREAD_AXES} dimensions, this kernel needs {axis + 1}",
            )
        return f"tid[{axis}]"

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        return f"{offset_var} + {self._tid(axis)} * {elements_per_thread}"

    def lane_offset_expr(self, lane_var: str) -> str:
        return lane_var

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"tgid[{dim}]"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {self._tid(axis)}"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {self._tid(axis)}"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = ({lid}) * ({block_size_var}) + {self._tid(axis)}"

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"{self._tid(axis)} < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        return True

    def launches_surplus_tile_threads(self) -> bool:
        # MPP requires num_warps * 32 threads on tid[0], which is independent
        # of the row block size; see the base docstring.
        return True

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from .metal_overrides import MetalOverrides

        return MetalOverrides()

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        metal_type = self.dtype_str(dtype)
        return f"{metal_type}({value_expr})"

    def reduction_acc_init_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # Must match the width metal/reduction.py casts the incoming value to,
        # or a rolled int8/bool reduction truncates (or saturates) each
        # iteration.  See _metal_acc_dtype.
        return f"{self.acc_type(dtype)}({value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return "0"

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        # Must be valid Python for expr_from_string; walker converts to C++ ternary
        return f"({true_val} if {mask} else {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"min({a}, {b})"

    def supports_config_key(self, key: str) -> bool:
        return key in self._SUPPORTED_CONFIG_KEYS

    # ------------------------------------------------------------------
    # Reductions.  See ``metal/reduction.py`` for the emitters and
    # ``metal/msl_reduction.py`` for the device-side helpers.
    # ------------------------------------------------------------------

    def max_reduction_threads(self) -> int | None:
        return MAX_THREADS_PER_THREADGROUP

    def reduction_axis_first(self) -> bool:
        # Metal linearizes ``tid`` with ``tid[0]`` fastest-varying and assigns
        # SIMD groups in that order, so putting the reduction on the lowest
        # axis is what makes its threads a contiguous run of SIMD lanes.
        return True

    def validate_reduction_input(self, block_index: int, value: torch.Tensor) -> None:
        """Reject a reduced value whose block id spans more than one axis.

        Equal-size reduction dimensions reuse one index. Metal maps that index
        to one thread axis, so broadcasting separately loaded values can make
        two logical axes collapse onto the same thread.
        """
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        axes = [
            dim
            for dim, size in enumerate(value.size())
            if env.get_block_id(size) == block_index
        ]
        if len(axes) > 1:
            raise exc.BackendUnsupported(
                self.name,
                f"reduction over a value whose dimensions {axes} share one "
                "index (equal-size reduction dimensions reuse a block id); "
                "Metal maps one index to one thread axis, so the axes would "
                "collapse onto the same thread",
            )

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        """Shrink the reduction span until the whole threadgroup fits in 1024.

        Halving preserves the power-of-two span that the SIMD butterflies and
        the two-stage threadgroup reduction both require.
        """
        from ..reduction_strategy import ReductionStrategy

        if requested <= 1:
            return requested
        other_threads = 1
        for strategy in existing_strategies:
            if isinstance(strategy, ReductionStrategy):
                count = strategy._reduction_thread_count()
                if count > 0:
                    other_threads *= count
            else:
                for size in strategy.thread_block_sizes():
                    if size > 1:
                        other_threads *= size
        while other_threads * requested > MAX_THREADS_PER_THREADGROUP and requested > 1:
            requested //= 2
        return requested

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        self._reject_extra_reduction_dimension()
        return self._tid(axis)

    def _reject_extra_reduction_dimension(self) -> None:
        """Allow at most one reduction dimension to hold a thread axis.

        Each reduction dimension claims a ``tid`` axis, but a second one is not
        given an axis disjoint from the tile strategy's: it lands back on an
        axis a tile already owns, so the two indices alias and the kernel reads
        (or writes) a diagonal.  Reject that rather than emit it.

        This is about distinct reduction *dimensions*, not distinct reductions:
        softmax's max and sum, or LayerNorm's mean and variance, share one
        dimension and one axis, and are unaffected.
        """
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        rdims = [bs.block_id for bs in env.block_sizes if bs.reduction]
        if len(rdims) > 1:
            raise exc.BackendUnsupported(
                self.name,
                f"{len(rdims)} reduction dimensions in one kernel "
                f"(block ids {rdims}); Metal gives each reduction dimension a "
                "thread axis and only one such axis is available alongside the "
                "tile axes",
            )

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "0"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"helion.next_power_of_2({expr})"

    def reduction_threads_hint(self, block_size_var: str | None = None) -> int | None:
        from ..device_function import DeviceFunction
        from .reduction import resolve_group

        group = resolve_group(DeviceFunction.current(), block_size_var)
        return group.span if group is not None else None

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        from .reduction import reduction_combine_expr

        return reduction_combine_expr(reduction_type, acc, val, dtype)

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> str:
        from .reduction import reduction_expr

        return reduction_expr(
            input_name,
            reduction_type,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
            dtype=dtype,
        )

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
        threads_in_group: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> str:
        from .reduction import argreduce_result_expr

        return argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            output_dtype,
            block_size_var=block_size_var,
            index_dtype=index_dtype,
            threads_in_group=threads_in_group,
            dtype=dtype,
        )

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
        dtype: torch.dtype | None = None,
    ) -> list[str]:
        from .reduction import argreduce_loop_update_statements

        return argreduce_loop_update_statements(
            reduction_type=reduction_type,
            acc=acc,
            acc_index=acc_index,
            value=value,
            index=index,
            dtype=dtype,
        )

    def create_reduction_strategy(
        self, fn: DeviceFunction, block_id: int, reduction_loop: int | None
    ) -> TileStrategy:
        """Build the reduction strategy and reject extents we cannot cover.

        Metal reduces one element per thread: unlike CuTe, it has no compiler
        machinery to carry a reduction across a lane loop, so a reduction is
        only correct when every element it has to cover in one pass is backed
        by a live thread.  Short of that it does not fail -- it reduces the
        first ``span`` elements and returns a plausible wrong answer.

        Both strategies can land there, for different reasons, so both are
        checked:

        - Persistent covers the whole dimension in one pass.  Dimensions too
          wide for a threadgroup are normally rolled by the config spec, but
          one the roller rejected (e.g. fused with a matmul) stays persistent.
        - Looped covers one rolled chunk per iteration.  The chunk is sized
          from ``reduction_loops``, then ``adjust_reduction_thread_count``
          shrinks the span to fit whatever the tile axes left over -- and can
          shrink it below the chunk.

        The budget passes in ``create_loop_strategy`` should keep the
        product within one threadgroup, making this unreachable; it is
        checked rather than assumed.
        """
        from ..compile_environment import CompileEnvironment
        from ..reduction_strategy import LoopedReductionStrategy
        from ..reduction_strategy import PersistentReductionStrategy
        from ..reduction_strategy import ReductionStrategy

        strategy = super().create_reduction_strategy(fn, block_id, reduction_loop)
        # Narrow for the type checker as CuTe does: only reduction strategies
        # carry a thread count; anything else counts zero and returns below.
        span = (
            strategy._reduction_thread_count()
            if isinstance(strategy, ReductionStrategy)
            else 0
        )
        if span <= 0:
            return strategy

        env = CompileEnvironment.current()
        if isinstance(strategy, PersistentReductionStrategy):
            covered = env.block_sizes[block_id].size_hint()
            what = f"persistent reduction over {covered} elements"
            hint = (
                "the tile axes left too few threads for it; give the reduction "
                "a smaller chunk with reduction_loops, or the tiles a smaller "
                "block_size"
            )
        elif isinstance(strategy, LoopedReductionStrategy):
            covered = strategy.block_size
            what = f"reduction loop over chunks of {covered} elements"
            hint = (
                "either the chunk exceeds a threadgroup or the tile axes left "
                "too few threads for it"
            )
        else:
            return strategy

        needed = self.static_rdim_size(covered)
        if needed > span:
            raise exc.BackendUnsupported(
                self.name,
                f"{what} needs {needed} threads but only {span} are available; {hint}",
            )
        return strategy

    def supports_precompile(self) -> bool:
        return False

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        """Wrap scalar SymbolArguments as 1-element tensors for buffer passing."""
        from ..device_function import SymbolArgument

        if isinstance(arg, SymbolArgument):
            device_expr = (
                f"{tensor_host_args[0]}.device" if tensor_host_args else "'mps'"
            )
            return (
                f"torch.scalar_tensor(float({host_str}), "
                f"dtype=torch.float32, "
                f"device={device_expr})"
            )
        return host_str

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from ..device_function import DeviceFunction

        dims = tuple(DeviceFunction.current().codegen.max_thread_block_dims)
        return [f"_block_dims=({dims[0]}, {dims[1]}, {dims[2]})"]

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
    ) -> list[str]:
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        """Metal loop strategy: reuse CuTe's planner.

        Metal and CuTe share the same scalar-thread execution model (one
        element per thread, cooperative hardware primitives for matmul), so
        they use the same ``PerThreadND``/``PerThreadFlattenedTileStrategy``
        with the same thread budget management, inactive block ID filtering,
        and auto-capping.

        Both paths treat an over-subscribed threadgroup the same way: a config
        that leaves every ``num_threads`` on auto is capped to 1024 and covers
        the remainder with a lane loop, while one that asks for more than 1024
        threads explicitly is rejected.  Users hitting that rejection should
        lower ``num_threads`` or ``block_sizes``.
        """
        config = self._config_with_mpp_thread_budget(fn, block_ids, config)
        # Cap tile thread counts so the reduction axes still fit alongside them.
        config = self._config_with_reduction_thread_budget(block_ids, config)
        # pyrefly: ignore[bad-argument-type]
        strategy = CuteBackend.create_loop_strategy(self, fn, block_ids, config)
        self._reject_dynamic_thread_extents(strategy)
        return strategy

    def _reject_dynamic_thread_extents(self, strategy: TileStrategy) -> None:
        """Reject a thread axis whose extent Helion cannot resolve statically.

        A dropped extent launches the axis one thread wide while its index stays
        ``offset + tid``, leaving most of the output silently wrong.  Removable
        once ``launcher_keyword_args`` takes the launch shape from the shared
        ``thread_block_dim_exprs`` rather than baking literal ``_block_dims``.
        """
        from ..tile_strategy import PerThreadFlattenedTileStrategy
        from ..tile_strategy import PerThreadNDTileStrategy

        if isinstance(strategy, PerThreadNDTileStrategy):
            dynamic = [
                block_size
                for block_id, block_size in zip(
                    strategy.block_ids, strategy.block_size, strict=True
                )
                if strategy._uses_thread_axis_for_block(block_id, block_size)
                and strategy._static_thread_extent_for_block(block_id, block_size)
                is None
            ]
        elif isinstance(strategy, PerThreadFlattenedTileStrategy):
            dynamic = []
            if strategy._uses_thread_axis() and not isinstance(
                strategy._thread_extent(), int
            ):
                dynamic.append(strategy.block_size)
        else:
            return
        if dynamic:
            raise exc.BackendUnsupported(
                self.name,
                f"thread extent {dynamic[0]} is not known until the kernel "
                "runs, and Helion fixes the Metal threadgroup shape at codegen "
                "time; give the tile a block_size the config can resolve, or "
                "compile the kernel with static_shapes=True",
            )

    def _reserved_reduction_threads(self, config: Config) -> int:
        """Threads the reduction axes will claim from the threadgroup.

        Every element of a persistent reduction has to be backed by a live
        thread -- Metal has no CuTe-style lane-loop fallback to cover a wider
        dimension with a narrower thread group.  Reduction dimensions too wide
        for one threadgroup are rolled by the config spec
        (``reduction_loops``), so the span needed here is the rolled chunk.
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        reserved = 1
        for block_size_info in env.block_sizes:
            if not block_size_info.reduction:
                continue
            block_id = block_size_info.block_id
            rolled = env.config_spec.reduction_loops.config_get(
                config.reduction_loops, block_id, None
            )
            extent = rolled if rolled is not None else block_size_info.size_hint()
            reserved *= next_power_of_2(
                max(1, min(int(extent), MAX_THREADS_PER_THREADGROUP))
            )
        return min(reserved, MAX_THREADS_PER_THREADGROUP)

    def _config_with_reduction_thread_budget(
        self, block_ids: list[int], config: Config
    ) -> Config:
        """Cap auto tile thread counts so the reduction axes still fit.

        Tile strategies are built before reduction strategies
        (``TileStrategyDispatch.__init__``), so without this the tiles claim
        the whole 1024-thread budget and ``adjust_reduction_thread_count``
        would shrink the reduction below the extent it has to cover.
        """
        reserved = self._reserved_reduction_threads(config)
        if reserved <= 1:
            return config

        from ...runtime.config import Config
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        num_threads = list(config.num_threads)
        if len(num_threads) < len(env.config_spec.num_threads):
            num_threads.extend(
                [0] * (len(env.config_spec.num_threads) - len(num_threads))
            )

        used = reserved
        changed = False
        tunable = set(env.config_spec.num_threads.valid_block_ids())
        for block_id in block_ids:
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            if configured > 0:
                used *= configured
                continue
            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue
            if block_id not in tunable:
                # An explicit ``hl.tile(n, block_size=<int>)`` or an ``hl.grid``
                # allocates no ``NumThreadsSpec``, so the axis has no
                # thread-count knob and always runs one thread per element.
                # Charge it to the budget rather than indexing a map it is not
                # in; if the total no longer fits, the reduction strategy
                # rejects the config with a diagnostic.
                used *= axis_size
                continue
            budget = max(1, MAX_THREADS_PER_THREADGROUP // max(1, used))
            chosen = _largest_divisor_at_most(axis_size, budget)
            if chosen >= axis_size:
                used *= chosen
                continue
            config_index = env.config_spec.num_threads.block_id_to_index(block_id)
            if num_threads[config_index] != chosen:
                num_threads[config_index] = chosen
                changed = True
            used *= chosen

        if not changed:
            return config
        return Config.from_dict({**config.config, "num_threads": num_threads})

    def _config_with_mpp_thread_budget(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> Config:
        """Reserve root-grid thread budget for MPPGraph cooperative work.

        MPP matmul and ordinary scalar Metal code run inside one Metal
        threadgroup.  MPP needs ``num_warps * 32`` threads participating on
        ``tid[0]`` for its cooperative operation, while scalar code in the
        surrounding root graph may still use ``tid[0]``, ``tid[1]``, and
        ``tid[2]`` for normal tile indexing.  This method keeps the root graph
        scalar-lowered, but caps auto ``num_threads`` on later root-grid axes
        so the combined threadgroup stays within Metal's 1024-thread limit.
        """
        if not any(
            type(graph_info).__name__ == "MPPGraphInfo"
            for graph_info in fn.codegen.codegen_graphs
        ):
            return config

        from ..host_function import HostFunction

        device_ir = HostFunction.current().device_ir
        # Only adjust the loop strategy for the root grid.  MPPGraphInfo emits
        # the cooperative K-loop internally; nested/device loops should keep
        # their normal Metal/CuTe strategy.
        if not device_ir.grid_block_ids or block_ids != device_ir.grid_block_ids[0]:
            return config
        if len(block_ids) < 2:
            return config

        from ...runtime.config import Config
        from ..compile_environment import CompileEnvironment
        from ..cute.thread_budget import MAX_THREADS_PER_BLOCK

        env = CompileEnvironment.current()
        num_threads = list(config.num_threads)
        if len(num_threads) < len(env.config_spec.num_threads):
            num_threads.extend(
                [0] * (len(env.config_spec.num_threads) - len(num_threads))
            )

        first_block_id = block_ids[0]
        first_axis_size = env.block_sizes[first_block_id].from_config(config)
        if not isinstance(first_axis_size, int):
            return config
        first_axis_configured = int(
            env.config_spec.num_threads.config_get(
                config.num_threads, first_block_id, 0
            )
        )
        first_axis_threads = (
            first_axis_configured if first_axis_configured > 0 else first_axis_size
        )

        # MPP's execution_simdgroups<N> uses N simdgroups, and each Metal
        # simdgroup has 32 threads.  tid[0] must be large enough for both
        # MPP's cooperative operation and any scalar indexing on the first
        # root axis.
        mpp_threads = config.num_warps * 32
        used_threads = max(mpp_threads, first_axis_threads)
        changed = False

        # Walk the remaining axes in launch order.  Explicit num_threads
        # consume budget as-is; auto axes are reduced to the largest divisor
        # that keeps the total threadgroup size under Metal's limit.
        for block_id in block_ids[1:]:
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            if configured > 0:
                used_threads *= configured
                continue

            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue

            budget = max(1, MAX_THREADS_PER_BLOCK // max(1, used_threads))
            chosen = _largest_divisor_at_most(axis_size, budget)
            config_index = env.config_spec.num_threads.block_id_to_index(block_id)
            if num_threads[config_index] != chosen:
                num_threads[config_index] = chosen
                changed = True
            used_threads *= chosen

        if not changed:
            return config
        return Config.from_dict({**config.config, "num_threads": num_threads})
