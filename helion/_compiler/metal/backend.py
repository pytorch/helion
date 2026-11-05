"""MetalBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

import torch

from ... import exc
from ..backend import Backend
from ..backend import BackendState
from ..backend import _largest_divisor_at_most
from ..cute.backend import CuteBackend
from ._constants import MAX_THREAD_AXES
from ._constants import MAX_THREADS_PER_THREADGROUP

if TYPE_CHECKING:
    from collections.abc import Callable

    import sympy
    from torch._inductor.ops_handler import OpsHandler

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..aten_lowering import Lowering
    from ..compile_environment import CompileEnvironment
    from ..device_function import Argument
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import BlockReductionStrategy
    from ..tile_dispatch import TileStrategyDispatch
    from ..tile_strategy import TileStrategy
    from .free_arange import FreeArangePlan
    from .mpp_matmul import MppPlan

    InductorOpOverrides = OpsHandler[Any]


def _reduction_contracted(block_id: int) -> bool:
    """Whether a reduction combines values across *block_id*.

    Read from the occurrence-built kernel facts: the reduction roller also
    tags slice-only dimensions with reduction loops, so loop structure
    cannot tell them apart.
    """
    from ..compile_environment import CompileEnvironment
    from .padded_twin import padded_twins

    reductions = CompileEnvironment.current().config_spec.reduction_kernel_fact
    # Built for every kernel before codegen, empty when it has no reductions.
    assert reductions is not None
    # A twin's reductions run on its owner's lanes (``MetalPaddedTwinStrategy``).
    blocks = {block_id, *padded_twins(block_id)}
    return any(reduction.block_id in blocks for reduction in reductions.reductions)


def _is_threadgroup_too_large(err: BaseException) -> bool:
    """Recognize Metal's pipeline-state error for an oversized threadgroup."""
    return "exceeds the maximum total threads per threadgroup" in str(err)


#: Wall-clock ceiling for a Metal autotune run, unless the caller sets
#: ``autotune_budget_seconds`` or asks for ``autotune_effort="full"``.  Every
#: candidate pays an MSL compile, so an unbounded search is slow enough to be
#: surprising.
_DEFAULT_AUTOTUNE_BUDGET_SECONDS = 300


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
            "reduction_fusion",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        # These are reset by ``pre_codegen`` and read after the codegen of the
        # same config, by the lowerings and ``launcher_keyword_args``:
        # pre_codegen, codegen and the launcher call run in that order for one
        # config at a time.
        self.free_arange_plan: FreeArangePlan = {}
        self.mpp_plan: MppPlan | None = None
        self._tid_axes: set[int] = set()
        # Set by ``pre_codegen`` (see ``_uniform_thread_extent``), and cleared
        # by ``create_loop_strategy`` because strategies of the next config
        # are built, and may read ``force_tile_mask``, before it runs.
        self._uniform_extent: int | None = None

    @property
    def name(self) -> str:
        return "metal"

    @property
    def state(self) -> BackendState:
        return BackendState.EXPERIMENTAL

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
    def supports_eager_prepared_call(self) -> bool:
        """Metal does nothing at call time before its generated wrapper.

        Output allocation and the launch both happen inside the wrapper,
        which the prepared call still enters.
        """
        return True

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
                "from helion._compiler.metal.metal_launcher"
                " import default_metal_launcher as _default_metal_launcher"
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

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .printer import metal_texpr

        return metal_texpr(expr)

    def thread_index(self, axis: int) -> str:
        """The ``tid`` component for a thread axis, rejecting a fourth one.

        Metal provides three.  A kernel wanting more -- a 3-D tile whose
        innermost dimension is also fully sliced, so that it claims an axis of
        its own -- would otherwise index past the end of the ``uint3``, which
        compiles cleanly and returns garbage.

        Records the axis: it is the only source of ``tid`` in loop index code,
        so a free ``hl.arange`` axis it never handed out is the arange's own.
        """
        self._tid_axes.add(axis)
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
        return f"{offset_var} + {self.thread_index(axis)} * {elements_per_thread}"

    def lane_offset_expr(self, lane_var: str) -> str:
        return lane_var

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"tgid[{dim}]"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {self.thread_index(axis)}"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {self.thread_index(axis)}"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return (
            f"{offsets_var} = ({lid}) * ({block_size_var}) + {self.thread_index(axis)}"
        )

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"{self.thread_index(axis)} < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        # The shared per-tile rule judges surplus threads by the dispatch's
        # planned thread axes, which can differ from the ones Metal emits and
        # launches, so it only applies to a uniform launch.
        return self._uniform_extent is None

    def launches_surplus_tile_threads(self) -> bool:
        # MPP requires num_warps * 32 threads on tid[0], which is independent
        # of the row block size; see the base docstring.
        return True

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from .metal_overrides import MetalOverrides

        return MetalOverrides()

    def pre_inductor_lowering(self, node: torch.fx.Node) -> Lowering | None:
        from ..aten_lowering import squeeze_lowering

        if node.target in (torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dims):
            # Same semantics as squeeze.dim; routed before Inductor turns them
            # into a ReinterpretView, which Helion's Inductor lowering refuses.
            return squeeze_lowering
        return None

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

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .free_arange import plan_free_aranges
        from .mpp_matmul import plan_mpp_matmuls
        from .positional_rebind import reject_positional_rebinds
        from .reduction import plan_reduced_tile_axes
        from .repeated_blocks import plan_repeated_blocks

        reject_positional_rebinds(graphs)
        plan_repeated_blocks(graphs, tile_strategy)
        plan_reduced_tile_axes(graphs, tile_strategy)
        self._tid_axes = set()
        self.free_arange_plan = plan_free_aranges(graphs, config, tile_strategy)
        self.mpp_plan = plan_mpp_matmuls(graphs)
        self._uniform_extent = self._uniform_thread_extent(graphs, tile_strategy)

    def _uniform_thread_extent(
        self, graphs: list[GraphInfo], tile_strategy: TileStrategyDispatch
    ) -> int | None:
        """The one thread extent E of every strategy's thread axes, or
        ``None`` if their extents differ, or the kernel has an MPP op or a
        free ``hl.arange``.

        The launch is the largest extent of each emitted thread axis, so with
        a single E every launch axis is 1 or E (``launcher_keyword_args``
        asserts it), and no tile runs surplus threads on whichever axis the
        emitted code gives it.  Free aranges and the MPP graph pass claim
        threads that no strategy owns, and a cooperative MPP matmul result
        holds its tile in MPP's layout rather than one element per thread.
        """
        if self.free_arange_plan or self._has_mpp_op(graphs):
            return None
        extents = {
            extent
            for strategy in tile_strategy.strategies
            for extent in strategy.thread_block_sizes()
        }
        if len(extents) > 1:
            return None
        return max(extents, default=1)

    def _has_mpp_op(self, graphs: list[GraphInfo]) -> bool:
        from .mpp_graph_codegen import MPPGraphInfo

        assert self.mpp_plan is not None
        return bool(self.mpp_plan.by_matmul) or any(
            isinstance(graph, MPPGraphInfo) for graph in graphs
        )

    def reduction_axis_first(self) -> bool:
        # Metal linearizes ``tid`` with ``tid[0]`` fastest-varying and assigns
        # SIMD groups in that order, so putting the reduction on the lowest
        # axis is what makes its threads a contiguous run of SIMD lanes.
        return True

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
        # TileStrategyDispatch.thread_axis_for_strategy gives every threaded
        # reduction dimension its own axis, so their indices never alias.
        return self.thread_index(axis)

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

        # The thread count comes from the strategy that owns block_size_var.
        assert threads_in_group is None
        return reduction_expr(
            input_name,
            reduction_type,
            block_size_var=block_size_var,
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

        assert threads_in_group is None
        return argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            output_dtype,
            block_size_var=block_size_var,
            index_dtype=index_dtype,
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

        Metal reduces one element per thread, so a reduction is only correct
        when every element it has to cover in one pass is backed by a live
        thread, or by a lane the strategy folds before the cross-thread
        combine.  Short of that it does not fail -- it reduces the first
        ``span`` elements and returns a plausible wrong answer.

        Both strategies can land there, for different reasons, so both are
        checked:

        - Persistent covers the whole dimension in one pass.  Dimensions too
          wide for a threadgroup are normally rolled by the config spec, but
          one the roller rejected (e.g. fused with a matmul) stays persistent.
        - Looped covers one rolled chunk per iteration.  The chunk is sized
          from ``reduction_loops``, then ``adjust_reduction_thread_count``
          shrinks the span to fit whatever the tile axes left over -- and can
          shrink it below the chunk.

        A starved persistent dimension is instead walked in a per-thread lane
        loop (``MetalPersistentReductionStrategy.serialize_lanes``).  A
        dimension that is only sliced (rope's ``q[tile_b, :, tile_t, :half]``)
        needs no combine across threads; one a reduction contracts folds each
        thread's lanes first.  A dot's K folds too: the scalar ``hl.dot``
        lowering pairs the operands in a per-lane product and folds the lanes
        when it sums them (``matmul_ops``).  Thread counts, lane counts and
        tail masks are compile-time, set from the first call's extent, so a
        persistent dimension whose extent is only known at run time is
        rejected: a later, longer call would drop its tail.

        The budget passes in ``create_loop_strategy`` should keep the
        product within one threadgroup, making this unreachable; it is
        checked rather than assumed.
        """
        from ..compile_environment import CompileEnvironment
        from ..reduction_strategy import LoopedReductionStrategy
        from ..reduction_strategy import PersistentReductionStrategy
        from ..reduction_strategy import ReductionStrategy
        from .padded_twin import padded_twin_owner
        from .reduction_strategy import MetalPaddedTwinStrategy
        from .reduction_strategy import MetalPersistentReductionStrategy

        owner = padded_twin_owner(block_id)
        if owner is not None:
            if reduction_loop is not None:
                raise exc.BackendUnsupported(
                    self.name,
                    f"reduction dim {block_id}, the power-of-two padding of dim "
                    f"{owner}, in a reduction loop; leave it persistent "
                    "(reduction_loops=None)",
                )
            return MetalPaddedTwinStrategy(fn, block_id, owner)
        if reduction_loop is None:
            size = CompileEnvironment.current().block_sizes[block_id].size
            if isinstance(size, torch.SymInt) and size.node.expr.free_symbols:
                raise exc.BackendUnsupported(
                    self.name,
                    f"a ':' dimension of runtime extent {size}: Metal fixes the "
                    "threads and lanes that cover it at codegen, from the first "
                    "call's extent, so a longer one would drop its tail; compile "
                    "the kernel with static_shapes=True, fix the size with "
                    "hl.specialize, or roll the dimension with reduction_loops "
                    "where the config offers it",
                )
            strategy = MetalPersistentReductionStrategy(fn, block_id)
        else:
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
            size = env.block_sizes[block_id].size
            if isinstance(strategy, MetalPersistentReductionStrategy) and isinstance(
                size, int
            ):
                # Lanes cover the real extent: covering its power-of-two
                # padding would only add fully masked lanes.
                strategy.serialize_lanes(size)
                if _reduction_contracted(block_id):
                    strategy.fold_lanes()
                return strategy
            raise exc.BackendUnsupported(
                self.name,
                f"{what} needs {needed} threads but only {span} are available; {hint}",
            )
        return strategy

    def create_block_reduction_strategy(
        self, state: CodegenState, block_id: int
    ) -> BlockReductionStrategy:
        from .reduction_strategy import MetalBlockReductionStrategy

        return MetalBlockReductionStrategy(state, block_id)

    # ------------------------------------------------------------------
    # Autotuning
    # ------------------------------------------------------------------

    def supports_precompile(self) -> bool:
        # There is no Triton-style out-of-process precompile for MSL; the
        # autotuner compiles and benchmarks each config inline.
        return False

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        # See metal/autotune.py for why neither shared benchmark works here.
        from .autotune import do_bench_metal

        return do_bench_metal

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        # Same rationale as get_do_bench, for the interleaved compare path.
        from .autotune import interleaved_bench_metal

        return interleaved_bench_metal

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # A config the backend cannot express (an unsupported reduction
        # layout, a threadgroup that does not fit) is an ordinary search miss:
        # log it and move on.  A shader that fails to compile is reported by
        # torch.mps as a SyntaxError and is almost always a real codegen bug,
        # so surface that at warn level without aborting the search.
        if isinstance(err, exc.BackendUnsupported):
            return "debug"
        if isinstance(err, RuntimeError) and _is_threadgroup_too_large(err):
            # Same category as BackendUnsupported -- the config simply does not
            # fit -- but Metal only notices when it builds the pipeline state,
            # long after codegen, so it arrives as a RuntimeError from the
            # launcher rather than a compile failure.  It is routine: 19% of a
            # sampled matmul space with explicit thread counts lands here, and
            # logging a fifth of the search at warn level would bury the
            # failures that do deserve attention.
            return "debug"
        if isinstance(err, SyntaxError):
            # torch.mps reports a shader that fails to compile as a
            # SyntaxError.  That is almost always a real codegen bug, so it is
            # surfaced without aborting the search.
            return "warn"
        # Anything else -- an MPS driver error, a pipeline-state failure we do
        # not have a pattern for -- is still just one bad candidate, but it is
        # not one we recognize, so say so.  Without a catch-all the shared
        # fallback is ``classify_triton_exception``, which returns "raise" for
        # any message it does not know; it knows no MPS message and Triton is
        # not even installed here, so an unfamiliar driver error would abort
        # the entire search.  Warn and move on, as the CuTe backend does --
        # and, like CuTe, leave non-``Exception`` BaseExceptions
        # (KeyboardInterrupt, SystemExit) to propagate.
        if isinstance(err, Exception):
            return "warn"
        return None

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        # Every candidate is compiled inline (no precompile) and MSL
        # compilation dominates, so bound the search unless the caller asked
        # for "full" or set an explicit budget.  Mirrors CuteBackend.autotune.
        settings = bound_kernel.settings
        original_budget = settings.autotune_budget_seconds
        if settings.autotune_budget_seconds is None and (
            settings.autotune_effort != "full"
        ):
            settings.autotune_budget_seconds = _DEFAULT_AUTOTUNE_BUDGET_SECONDS
        try:
            return super().autotune(bound_kernel, args, force=force, **kwargs)
        finally:
            settings.autotune_budget_seconds = original_budget

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from ..device_function import DeviceFunction
        from .free_arange import check_free_arange_launch
        from .mpp_matmul import check_mpp_launch

        dims = tuple(DeviceFunction.current().codegen.max_thread_block_dims)
        if self._uniform_extent is not None:
            assert all(dim in (1, self._uniform_extent) for dim in dims), (
                f"tile masks are elided only if every launch axis is 1 or "
                f"{self._uniform_extent}, but the kernel launches {dims}"
            )
        check_free_arange_launch(self.free_arange_plan, self._tid_axes, dims)
        if self.mpp_plan is not None:
            check_mpp_launch(self.mpp_plan, dims)
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
        from ..device_function import SymbolArgument

        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        keywords = self.launcher_keyword_args(config, has_barrier=has_barrier)
        by_value = tuple(
            i
            for i, arg in enumerate(sorted_args or ())
            if isinstance(arg, SymbolArgument)
        )
        if by_value:
            keywords.append(f"_by_value={by_value!r}")
        return [*args, *keywords]

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
        self._uniform_extent = None
        # Claim the MPP matmul's threads before the tiles take the whole budget.
        config = self._config_with_mpp_thread_budget(fn, block_ids, config)
        # Cap tile thread counts so the reduction axes and free hl.arange axes
        # still fit alongside them.
        config = self._config_with_reduction_thread_budget(block_ids, config)
        # Re-clamp any count the passes above left not dividing its block size.
        config = self._config_with_divisible_thread_counts(block_ids, config)
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

    def _mutable_num_threads(self, config: Config) -> list[int]:
        """Mutable copy of ``config.num_threads`` for the passes below.

        Extended with auto (``0``) sentinels to the spec length, so a
        config that leaves trailing axes unset can still be indexed by
        ``block_id_to_index``.
        """
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        num_threads = list(config.num_threads)
        if len(num_threads) < len(env.config_spec.num_threads):
            num_threads.extend(
                [0] * (len(env.config_spec.num_threads) - len(num_threads))
            )
        return num_threads

    def _record_thread_count(
        self,
        env: CompileEnvironment,
        num_threads: list[int],
        block_id: int,
        chosen: int,
    ) -> bool:
        """Write an explicit thread count; return True if it changed anything."""
        config_index = env.config_spec.num_threads.block_id_to_index(block_id)
        if num_threads[config_index] != chosen:
            num_threads[config_index] = chosen
            return True
        return False

    def _finalize_thread_config(
        self, config: Config, num_threads: list[int], changed: bool
    ) -> Config:
        """Return the original config when no axis changed, else a copy."""
        if not changed:
            return config
        from ...runtime.config import Config

        return Config.from_dict({**config.config, "num_threads": num_threads})

    def _config_with_divisible_thread_counts(
        self, block_ids: list[int], config: Config
    ) -> Config:
        """Round each explicit ``num_threads`` down to a divisor of its block size.

        The shared loop strategy requires ``block_size % num_threads == 0`` and
        raises otherwise.  Autotuning draws ``num_threads`` from the tensor
        extent rather than the chosen block size, so a meaningful slice of the
        search space would otherwise be spent on configs that cannot compile --
        and a batch that happens to contain only such configs aborts the whole
        search.  Clamping keeps every config in the space legal, in the same
        spirit as the two thread-budget passes above.
        """
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        num_threads = list(config.num_threads)
        changed = False
        for block_id in block_ids:
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            if configured <= 0:
                continue  # 0 means "use the block size"
            block_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(block_size, int) or block_size % configured == 0:
                continue
            chosen = _largest_divisor_at_most(block_size, configured)
            if self._record_thread_count(env, num_threads, block_id, chosen):
                changed = True
        return self._finalize_thread_config(config, num_threads, changed)

    def _reserved_reduction_threads(self, config: Config) -> int:
        """Threads the reduction axes will claim from the threadgroup.

        Every element of a persistent reduction should be backed by a live
        thread: only a static dimension can be walked in lanes instead
        (``MetalPersistentReductionStrategy.serialize_lanes``), at the cost
        of serializing it.  Reduction dimensions too wide for one threadgroup
        are rolled by the config spec (``reduction_loops``), so the span
        needed here is the rolled chunk.
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

    def _reserved_free_arange_threads(self, config: Config) -> int:
        """Threads the free ``hl.arange`` axes will claim: one per element of
        each free block (``metal/free_arange.py``)."""
        from .free_arange import free_arange_blocks
        from .free_arange import free_block_size

        return math.prod(
            free_block_size(block_id, config) for block_id in free_arange_blocks()
        )

    def _config_with_reduction_thread_budget(
        self, block_ids: list[int], config: Config
    ) -> Config:
        """Cap tile thread counts so the reduction axes and the free
        ``hl.arange`` axes still fit.

        Tile strategies are built before reduction strategies
        (``TileStrategyDispatch.__init__``), so without this the tiles claim
        the whole 1024-thread budget and ``adjust_reduction_thread_count``
        would shrink the reduction below the extent it has to cover, which
        ``create_reduction_strategy`` then rejects unless the dimension can
        be walked in lanes.

        Autotuning makes that a search-efficiency problem as well as a
        correctness one: it sets ``num_threads`` explicitly, so capping only
        the *auto* axes left a slice of the space unable to compile.  Explicit
        counts are capped too.

        A reservation over 1024 leaves every tile one thread, and
        ``check_free_arange_launch`` rejects the launch.
        """
        reserved = self._reserved_reduction_threads(config)
        reserved *= self._reserved_free_arange_threads(config)
        if reserved <= 1:
            return config

        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        num_threads = self._mutable_num_threads(config)

        used = reserved
        changed = False
        tunable = set(env.config_spec.num_threads.valid_block_ids())
        for block_id in block_ids:
            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue
            if block_id not in tunable:
                # An explicit ``hl.tile(n, block_size=<int>)`` or an ``hl.grid``
                # allocates no ``NumThreadsSpec``, so the axis has no
                # thread-count knob and always runs one thread per element.
                # Charge it to the budget rather than indexing a map it is not
                # in; if the total no longer fits, the reduction strategy
                # walks the reduction in lanes or rejects the config.
                used *= axis_size
                continue
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            # 0 means "use the block size"; anything larger than the block size
            # is capped there because the axis has no more work than that.
            requested = min(configured, axis_size) if configured > 0 else axis_size
            budget = max(1, MAX_THREADS_PER_THREADGROUP // max(1, used))
            chosen = _largest_divisor_at_most(axis_size, min(requested, budget))
            used *= chosen
            if configured == 0 and chosen >= axis_size:
                continue  # leave the auto sentinel alone
            if self._record_thread_count(env, num_threads, block_id, chosen):
                changed = True

        return self._finalize_thread_config(config, num_threads, changed)

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
        # Any root grid, not just the first: rewrite_mpp_graphs scans every
        # RootGraphInfo, so a matmul in a later top-level hl.tile still gets
        # MPP lowering and still needs its thread budget reserved.
        if block_ids not in device_ir.grid_block_ids:
            return config
        if len(block_ids) < 2:
            return config

        from ..compile_environment import CompileEnvironment
        from ..cute.thread_budget import MAX_THREADS_PER_BLOCK

        env = CompileEnvironment.current()
        num_threads = self._mutable_num_threads(config)

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

        # Walk the remaining axes in launch order.  Auto axes are reduced to the
        # largest divisor that keeps the total threadgroup size under Metal's
        # limit; an explicit count is taken as given, but is rejected if the
        # product cannot launch.  Metal only refuses an oversized threadgroup
        # when it builds the pipeline state, so without this the config costs a
        # full MSL compile before failing -- 15-45% of a sampled matmul search
        # space, depending on seed.
        for block_id in block_ids[1:]:
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            if configured > 0:
                used_threads *= configured
                if used_threads > MAX_THREADS_PER_BLOCK:
                    raise exc.BackendUnsupported(
                        self.name,
                        f"threadgroup of {used_threads} threads "
                        f"(max {MAX_THREADS_PER_BLOCK}); MPP claims "
                        f"{mpp_threads} on tid[0] and the explicit num_threads "
                        "on the remaining axes do not fit alongside it",
                    )
                continue

            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue

            budget = max(1, MAX_THREADS_PER_BLOCK // max(1, used_threads))
            chosen = _largest_divisor_at_most(axis_size, budget)
            if self._record_thread_count(env, num_threads, block_id, chosen):
                changed = True
            used_threads *= chosen

        return self._finalize_thread_config(config, num_threads, changed)
