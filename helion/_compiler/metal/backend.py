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
from ..ast_extension import convert
from ..backend import Backend
from ..backend import BackendState
from ..backend import _largest_divisor_at_most
from ._constants import MAX_THREAD_AXES
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .msl_expr import PURE_CALLS
from .msl_expr import cast_source
from .msl_expr import parse_msl_expression

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    import sympy
    from torch._inductor.ops_handler import OpsHandler

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..aten_lowering import Lowering
    from ..autotuner_heuristics.registry import AutotunerHeuristicType
    from ..compile_environment import CompileEnvironment
    from ..device_function import Argument
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
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


def threadgroup_budget(config: Config, device_ir: DeviceIR | None = None) -> int:
    """The threads a launch may use: ``num_warps`` SIMD groups, at most
    Metal's 1024, and at least what the reduction dims' lane forms need
    (``_lane_floor``).  Every block that takes a thread axis gets its threads
    within it; lanes walk the rest of its extent."""
    budget = min(MAX_THREADS_PER_THREADGROUP, SIMD_WIDTH * int(config.num_warps))
    return max(budget, min(MAX_THREADS_PER_THREADGROUP, _lane_floor(config, device_ir)))


def budget_note() -> str:
    """Suffix for a rejection that depends on the launch size: the budget the
    config's num_warps gave the kernel, when it is below Metal's 1024."""
    from ..compile_environment import CompileEnvironment

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    if backend._thread_budget >= MAX_THREADS_PER_THREADGROUP:
        return ""
    return (
        f" (threadgroup budget {backend._thread_budget} threads from "
        f"num_warps={backend._num_warps})"
    )


def indexed_reduction_blocks(device_ir: DeviceIR) -> set[int]:
    """Reduction dims an argmin/argmax contracts; these have no lane form
    (``MetalPersistentReductionStrategy.codegen_reduction``)."""
    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import ReductionLowering

    backend = CompileEnvironment.current().backend
    return {
        lowering.block_index
        for info in device_ir.graphs
        for node in info.graph.nodes
        if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
        and backend.is_indexed_reduction(lowering.reduction_type)
    }


def static_extent(size: object) -> int | None:
    """*size* as an int if codegen knows it: a literal, or a SymInt with no
    free symbols left (a dim ``hl.specialize`` pinned under dynamic shapes)."""
    if isinstance(size, int):
        return size
    if isinstance(size, torch.SymInt) and not size.node.expr.free_symbols:
        return int(size)
    return None


def _lane_floor(config: Config, device_ir: DeviceIR | None = None) -> int:
    """The fewest threads at which every persistent reduction dim fits its
    lane form beside the root tiles that walk no lanes: a fold walks at most
    ``MAX_LANE_COPIES`` lanes, while an argmin/argmax and an extent codegen
    doesn't know (``static_extent``) walk none."""
    from torch._inductor.runtime.runtime_utils import next_power_of_2

    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .lane_fold import MAX_LANE_COPIES
    from .padded_twin import padded_twin_owner

    env = CompileEnvironment.current()
    if device_ir is None:
        # Inside create_loop_strategy, which has just set the codegen graphs' set.
        device_ir = HostFunction.current().device_ir
        assert isinstance(env.backend, MetalBackend)
        scalar_dots = env.backend._scalar_contraction_blocks
    else:
        scalar_dots = scalar_contraction_blocks(device_ir.graphs, live=False)
    indexed = indexed_reduction_blocks(device_ir)
    # A literal-size root tile has no num_threads: it runs one thread per
    # element whatever the budget (_config_with_reduction_thread_budget).
    tunable = set(env.config_spec.num_threads.valid_block_ids())
    floor = 1
    for block_id in {b for ids in device_ir.grid_block_ids for b in ids} - tunable:
        size = env.block_sizes[block_id].from_config(config)
        if isinstance(size, int):
            floor *= size
    for info in env.block_sizes:
        if not info.reduction or padded_twin_owner(info.block_id) is not None:
            continue  # a padded twin walks its owner's lanes
        block_id = info.block_id
        if (
            env.config_spec.reduction_loops.config_get(
                config.reduction_loops, block_id, None
            )
            is not None
        ):
            continue  # a rolled chunk steps by its threads (create_reduction_strategy)
        extent = int(info.size_hint())
        if static_extent(info.size) is None or block_id in indexed:
            floor *= next_power_of_2(max(1, extent))
        elif _reduction_contracted(block_id) or block_id in scalar_dots:
            floor *= next_power_of_2(max(1, -(-extent // MAX_LANE_COPIES)))
    return floor


def blocks_reduced_after_loop(device_ir: DeviceIR) -> set[int]:
    """Tile blocks a reduction contracts outside the loop that walks them."""
    from ..device_ir import ForLoopGraphInfo
    from ..inductor_lowering import ReductionLowering

    out: set[int] = set()
    for info in device_ir.graphs:
        loop_blocks = info.block_ids if isinstance(info, ForLoopGraphInfo) else []
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and lowering.block_index not in loop_blocks
            ):
                out.add(lowering.block_index)
    return out


def scalar_contraction_blocks(graphs: list[GraphInfo], live: bool = True) -> set[int]:
    """K blocks of the live hl.dot ops whose K is not held as per-thread
    parts: those the scalar contraction lowers (``matmul_ops``) and the
    whole-K ones ``plan_mpp_matmuls`` takes, whose ``:`` M or N admission
    counts on the lane floor's threads for these blocks.  An aten matmul
    lowers through MPP or is rejected.  ``live=False`` scans every graph (no
    compiling function to tell live ones), which can only add blocks."""
    from ...language.matmul_ops import dot
    from ..compile_environment import CompileEnvironment
    from .factored_dims import factored_size
    from .free_arange import live_graphs

    env = CompileEnvironment.current()
    out: set[int] = set()
    for info in live_graphs(graphs) if live else graphs:
        for node in info.graph.find_nodes(op="call_function", target=dot):
            k = node.args[0].meta["val"].size(-1)
            if factored_size(k) is None:
                block_id = env.resolve_block_id(k)
                if block_id is not None:
                    out.add(block_id)
    return out


def _is_threadgroup_too_large(err: BaseException) -> bool:
    """Recognize Metal's pipeline-state error for an oversized threadgroup."""
    return "exceeds the maximum total threads per threadgroup" in str(err)


#: Wall-clock ceiling for a Metal autotune run, unless the caller sets
#: ``autotune_budget_seconds`` or asks for ``autotune_effort="full"``.  Every
#: candidate pays an MSL compile, so an unbounded search is slow enough to be
#: surprising.
_DEFAULT_AUTOTUNE_BUDGET_SECONDS = 300


def _config_fixed_scalars() -> set[str]:
    """Runtime scalar arguments whose expression has only block sizes as free
    symbols, e.g. a tile count ``cdiv(m, block_m)`` under static shapes: the
    config fixes their value."""
    from ..device_function import DeviceFunction
    from ..device_function import SymbolArgument
    from ..host_function import HostFunction
    from ..variable_origin import BlockSizeOrigin

    origins = HostFunction.current().expr_to_origin

    def is_block_size(sym: sympy.Basic) -> bool:
        info = origins.get(sym)
        return info is not None and isinstance(info.origin, BlockSizeOrigin)

    return {
        arg.name
        for expr, arg in DeviceFunction.current()._expr_args.items()
        if isinstance(arg, SymbolArgument)
        and expr.free_symbols
        and all(is_block_size(sym) for sym in expr.free_symbols)
    }


class MetalBackend(Backend):
    """Metal Shading Language (MSL) code generation backend for macOS."""

    def autotuner_heuristics(self) -> tuple[AutotunerHeuristicType, ...]:
        from .row_reduction_heuristic import MetalRowReductionHeuristic

        return (MetalRowReductionHeuristic,)

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
            "pid_type",
            "num_sm_multiplier",
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
        # The config's ``threadgroup_budget``, set by ``create_loop_strategy``
        # for the reduction strategies built after the loop strategies.
        self._thread_budget = MAX_THREADS_PER_THREADGROUP
        self._num_warps = MAX_THREADS_PER_THREADGROUP // SIMD_WIDTH
        self._scalar_contraction_blocks: set[int] = set()
        # Threads left for the current loop's literal-size tiles when the
        # kernel's tiles overflow the budget (_serialize_nested_auto_tiles).
        self._literal_tile_limit: int | None = None

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
        if dtype_str == "bfloat":
            # Metal's int -> bfloat conversion is ~3.4x slower than int ->
            # float -> bfloat, and rounds once where torch CPU and CUDA round
            # through float (c10's BFloat16 is built from a float only).
            # From a floating type the float step is exact.
            return cast_source("bfloat", cast_source("float", expr_str))
        return cast_source(dtype_str, expr_str)

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
        # Each launch axis takes the largest extent among the strategies that
        # emit it, so a smaller tile on that axis has surplus threads; see the
        # base docstring.
        return True

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from .metal_overrides import MetalOverrides

        return MetalOverrides()

    @property
    def pure_device_calls(self) -> frozenset[str]:
        return PURE_CALLS

    def parse_op_expression(self, text: str) -> ast.AST:
        # MetalOverrides writes MSL (``a && b``, ``::metal::precise::atan2``,
        # ``static_cast<T>(x)``), which is not Python.
        return convert(parse_msl_expression(text))

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

    def default_num_sm_multiplier(self) -> int:
        # A threadgroup doesn't fill an Apple GPU core.  On an M4 Pro (16
        # cores), persistent matmul and attention tiles ran about 3x slower at
        # one threadgroup per core than at four, 2 per core left most of the
        # matmul gain on the table, and 8 slowed memory-bound kernels.
        return 4

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
        from ..compile_environment import CompileEnvironment
        from .free_arange import check_free_lanes
        from .free_arange import plan_free_aranges
        from .literal_dims import reject_literal_dims
        from .mpp_matmul import plan_mpp_matmuls
        from .positional_rebind import reject_positional_rebinds
        from .reduction import plan_reduced_tile_axes
        from .repeated_blocks import plan_repeated_blocks

        reject_positional_rebinds(graphs)
        reject_literal_dims(graphs, config)
        plan_repeated_blocks(graphs, tile_strategy)
        plan_reduced_tile_axes(graphs, tile_strategy)
        self._tid_axes = set()
        self.free_arange_plan = plan_free_aranges(graphs, config, tile_strategy)
        threaded = {
            block
            for block in range(len(CompileEnvironment.current().block_sizes))
            if (tile_strategy.thread_extent_for_block_id(block) or 1) > 1
        }
        self.mpp_plan = plan_mpp_matmuls(graphs, threaded)
        check_free_lanes(self.free_arange_plan, self.mpp_plan, graphs)
        self._uniform_extent = self._uniform_thread_extent(tile_strategy)

    def _uniform_thread_extent(self, tile_strategy: TileStrategyDispatch) -> int | None:
        """The one thread extent E of every strategy's thread axes, or
        ``None`` if their extents differ, or the kernel has a free
        ``hl.arange``.

        The launch is the largest extent of each emitted thread axis, so with
        a single E every launch axis is 1 or E (``launcher_keyword_args``
        asserts it), and no tile runs surplus threads on whichever axis the
        emitted code gives it.  Free aranges claim threads that no strategy
        owns.  The MPP planner launches only the strategies' threads, and an
        element of a cooperative result is visited at its row and column,
        which lie inside the op's tile and so inside each block: the shared
        rule's masks hold there too.
        """
        if self.free_arange_plan:
            return None
        extents = {
            extent
            for strategy in tile_strategy.strategies
            for extent in strategy.thread_block_sizes()
        }
        if len(extents) > 1:
            return None
        return max(extents, default=1)

    def reduction_axis_first(self) -> bool:
        # Metal linearizes ``tid`` with ``tid[0]`` fastest-varying and assigns
        # SIMD groups in that order, so putting the reduction on the lowest
        # axis is what makes its threads a contiguous run of SIMD lanes.
        return True

    def matmul_contraction_thread_reserve(
        self, fn: DeviceFunction, tile_block_ids: list[int]
    ) -> int:
        """Not a contraction reserve on Metal: this hook carries the thread
        limit for literal-size tiles into the shared planner.

        The shared planner reserves a matmul's whole contraction extent
        because CuTe's scalar contraction can't walk it in lanes.  On Metal an
        MPP op contracts its own K, a scalar hl.dot folds its K lanes
        (matmul_ops), and the dimensions a reduction combines are reserved by
        ``_config_with_reduction_thread_budget``, so the reserve is 1.

        A literal-size tile has no ``num_threads`` for the Metal passes to
        record.  When the kernel's tiles overflow what the reservations leave,
        ``_serialize_nested_auto_tiles`` sets ``_literal_tile_limit``; this
        returns ``1024 / limit``, and the shared planner's split (OPTION B)
        shrinks the call's auto tiles to ``1024 / reserve``.  The limit is
        rounded down to a power of two so that round trip is exact.  The
        planner only ever takes the smaller of this and its own limit (which
        is 256 on its matmul branch), so the hook can shrink tiles, never
        widen them.
        """
        if self._literal_tile_limit is not None:
            limit = 1 << (self._literal_tile_limit.bit_length() - 1)
            return MAX_THREADS_PER_THREADGROUP // limit
        return 1

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        """Shrink the reduction span until the whole threadgroup fits the
        config's budget (``threadgroup_budget``).

        Halving preserves the power-of-two span that the SIMD butterflies and
        the two-stage threadgroup reduction both require.
        """
        from ..reduction_strategy import ReductionStrategy

        budget = self._thread_budget

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
        while other_threads * requested > budget and requested > 1:
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
        from torch._inductor.runtime.runtime_utils import next_power_of_2

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
            if isinstance(size, torch.SymInt) and static_extent(size) is None:
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
            # A rolled chunk wider than the threads the budget leaves it steps
            # by those threads: thread t visits elements t, t + span, ..., the
            # order a lane loop over the chunk would give it.
            span = self.adjust_reduction_thread_count(
                next_power_of_2(min(reduction_loop, MAX_THREADS_PER_THREADGROUP)),
                fn.tile_strategy.strategies,
            )
            if 1 < span < reduction_loop:
                reduction_loop = span
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
            extent = static_extent(env.block_sizes[block_id].size)
            if (
                isinstance(strategy, MetalPersistentReductionStrategy)
                and extent is not None
            ):
                # Lanes cover the real extent: covering its power-of-two
                # padding would only add fully masked lanes.
                strategy.serialize_lanes(extent)
                if _reduction_contracted(block_id):
                    strategy.fold_lanes()
                return strategy
            raise exc.BackendUnsupported(
                self.name,
                f"{what} needs {needed} threads but only {span} are available; "
                f"{hint}{budget_note()}",
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
        fixed = _config_fixed_scalars()
        by_value = tuple(
            i
            for i, arg in enumerate(sorted_args or ())
            if isinstance(arg, SymbolArgument) and arg.name not in fixed
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
        self._literal_tile_limit = None
        self._scalar_contraction_blocks = scalar_contraction_blocks(
            fn.codegen.codegen_graphs
        )
        self._thread_budget = threadgroup_budget(config)
        self._num_warps = int(config.num_warps)
        # Cap tile thread counts so the reduction axes and free hl.arange axes
        # still fit alongside them.
        config = self._config_with_reduction_thread_budget(fn, block_ids, config)
        # Re-clamp any count the passes above left not dividing its block size.
        config = self._config_with_divisible_thread_counts(block_ids, config)
        from ..per_thread_planning import plan_per_thread_loop_strategy

        strategy = plan_per_thread_loop_strategy(self, fn, block_ids, config)
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

    def _reserved_reduction_threads(
        self,
        config: Config,
        combined: bool = False,
        device_ir: DeviceIR | None = None,
    ) -> int:
        """Threads the reduction axes will claim from the threadgroup.

        Every element of a persistent reduction should be backed by a live
        thread: only a static dimension can be walked in lanes instead
        (``MetalPersistentReductionStrategy.serialize_lanes``), at the cost
        of serializing it.  Reduction dimensions too wide for one threadgroup
        are rolled by the config spec (``reduction_loops``), so the span
        needed here is the rolled chunk.

        ``combined`` reserves only the dimensions whose values Scalar code
        combines, each its share of a full threadgroup scaled to the budget,
        and the ones that can't walk lanes; without it every reduction
        dimension reserves its whole extent up to the budget.
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction

        env = CompileEnvironment.current()
        device_ir = device_ir or HostFunction.current().device_ir
        budget = threadgroup_budget(config, device_ir)
        indexed = indexed_reduction_blocks(device_ir)
        reserved = 1
        for block_size_info in env.block_sizes:
            if not block_size_info.reduction:
                continue
            block_id = block_size_info.block_id
            rolled = env.config_spec.reduction_loops.config_get(
                config.reduction_loops, block_id, None
            )
            extent = int(rolled if rolled is not None else block_size_info.size_hint())
            # A rolled chunk steps by its threads (create_reduction_strategy), a
            # persistent dim walks lanes only over a static extent, and an
            # argmin/argmax has no lane form.
            can_lane = (
                rolled is not None or static_extent(block_size_info.size) is not None
            ) and block_id not in indexed
            full = next_power_of_2(max(1, min(extent, budget)))
            if not combined or not can_lane:
                reserved *= full
                continue
            if (
                not _reduction_contracted(block_id)
                and block_id not in self._scalar_contraction_blocks
            ):
                # A sliced dimension walks lanes as a tile does and takes what
                # the tiles leave (adjust_reduction_thread_count).
                continue
            # A looped reduction needs two threads (LoopedReductionStrategy).
            least = min(2, next_power_of_2(extent)) if rolled is not None else 1
            share = next_power_of_2(extent) * budget // MAX_THREADS_PER_THREADGROUP
            reserved *= min(full, max(least, share))
        return min(reserved, budget)

    def _reserved_free_arange_threads(self, config: Config) -> int:
        """Threads the free ``hl.arange`` axes will claim
        (``free_arange.free_block_threads``)."""
        from .free_arange import free_block_threads

        return math.prod(threads for threads, _ in free_block_threads(config).values())

    def _config_with_reduction_thread_budget(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> Config:
        """Fit the tiles' thread counts to the threadgroup budget
        (``threadgroup_budget``) beside the reduction axes and the free
        ``hl.arange`` axes.

        The reduction axes Scalar code combines reserve their share first
        (``_reserved_reduction_threads``); the tiles then take, in block order,
        the largest divisor of their requested count (explicit or auto) that
        fits what is left; the reductions fill the rest
        (``adjust_reduction_thread_count``).  Tile strategies are built before
        reduction strategies (``TileStrategyDispatch.__init__``), hence the
        reservation.

        An explicit ``hl.tile(n, block_size=<int>)`` or an ``hl.grid``
        allocates no ``NumThreadsSpec``: it runs one thread per element and is
        charged as such.
        """
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        budget = threadgroup_budget(config)
        tunable = set(env.config_spec.num_threads.valid_block_ids())
        reserved = self._reserved_reduction_threads(config, combined=True)
        reserved *= self._reserved_free_arange_threads(config)
        if reserved <= 1:
            return self._config_with_auto_threads_shrunk(fn, block_ids, config, budget)

        num_threads = self._mutable_num_threads(config)

        used = reserved
        changed = False
        for block_id in block_ids:
            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue
            if block_id not in tunable:
                used *= axis_size
                continue
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            # 0 means "use the block size"; anything larger than the block size
            # is capped there because the axis has no more work than that.
            requested = min(configured, axis_size) if configured > 0 else axis_size
            budget_left = max(1, budget // max(1, used))
            chosen = _largest_divisor_at_most(axis_size, min(requested, budget_left))
            used *= chosen
            if configured == 0 and chosen >= axis_size:
                continue  # leave the auto sentinel alone
            if self._record_thread_count(env, num_threads, block_id, chosen):
                changed = True
        changed |= self._serialize_nested_auto_tiles(
            fn, block_ids, config, num_threads, budget // reserved
        )
        return self._finalize_thread_config(config, num_threads, changed)

    def _serialize_nested_auto_tiles(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        config: Config,
        num_threads: list[int],
        budget: int,
    ) -> bool:
        """When the kernel's loop blocks together pass ``budget``, shrink this
        device loop's auto tiles to the largest divisor that fits; lanes walk
        the rest.  A tile reduced after its loop needs one element per thread
        or one thread (``MetalBlockReductionStrategy``), so it goes to one."""
        from ..backend import _active_loop_block_ids
        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction

        env = CompileEnvironment.current()
        spec = env.config_spec.num_threads
        tunable = set(spec.valid_block_ids())
        total = 1
        for block_id in _active_loop_block_ids(fn):
            info = env.block_sizes[block_id]
            size = info.from_config(config)
            if info.reduction or not isinstance(size, int):
                continue
            threads = (
                num_threads[spec.block_id_to_index(block_id)]
                if block_id in tunable
                else 0
            )
            total *= threads if threads > 0 else size
        if total <= budget:
            return False
        device_ir = HostFunction.current().device_ir
        grid_ids = {b for ids in device_ir.grid_block_ids for b in ids}
        after_loop = blocks_reduced_after_loop(device_ir)
        changed = False
        for block_id in block_ids:
            if total <= budget:
                break
            if block_id not in tunable or block_id in grid_ids:
                continue
            if int(spec.config_get(config.num_threads, block_id, 0)) != 0:
                continue
            size = env.block_sizes[block_id].from_config(config)
            if not isinstance(size, int):
                continue
            # The tile pass may already have given the block fewer threads
            # than its size; that count is its share of ``total``.
            current = num_threads[spec.block_id_to_index(block_id)] or size
            rest = total // current
            if block_id in after_loop:
                chosen = 1
            else:
                chosen = _largest_divisor_at_most(
                    size, min(current, max(1, budget // rest))
                )
            total = rest * chosen
            changed |= self._record_thread_count(env, num_threads, block_id, chosen)
        literal = math.prod(
            size
            for block_id in block_ids
            if block_id not in tunable
            and block_id not in grid_ids
            and isinstance(size := env.block_sizes[block_id].from_config(config), int)
        )
        if total > budget and literal > 1:
            self._literal_tile_limit = max(1, budget // max(1, total // literal))
        return changed

    def _config_with_auto_threads_shrunk(
        self, fn: DeviceFunction, block_ids: list[int], config: Config, budget: int
    ) -> Config:
        """With no reduction or free axis, the shared planner shrinks auto
        tiles to 1024 threads, largest first; shrink them the same way to a
        budget below that, explicit counts included."""
        from ..compile_environment import CompileEnvironment
        from ..host_function import HostFunction

        env = CompileEnvironment.current()
        tunable = set(env.config_spec.num_threads.valid_block_ids())
        num_threads = self._mutable_num_threads(config)
        fixed = 1
        counts: dict[int, tuple[int, int]] = {}
        for block_id in block_ids:
            axis_size = env.block_sizes[block_id].from_config(config)
            if not isinstance(axis_size, int):
                continue
            if block_id not in tunable:
                fixed *= axis_size
                continue
            configured = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            requested = min(configured, axis_size) if configured > 0 else axis_size
            counts[block_id] = (axis_size, requested)
        threads = {b: r for b, (_, r) in counts.items()}
        after_loop = blocks_reduced_after_loop(HostFunction.current().device_ir)
        while fixed * math.prod(threads.values()) > budget:
            block_id = max(threads, key=lambda b: threads[b], default=None)
            if block_id is None or threads[block_id] <= 1:
                break
            # A tile reduced after its loop needs one element per thread or
            # one thread (``MetalBlockReductionStrategy``).
            threads[block_id] = (
                1
                if block_id in after_loop
                else _largest_divisor_at_most(
                    counts[block_id][0], threads[block_id] - 1
                )
            )
        changed = False
        for block_id, chosen in threads.items():
            if chosen == counts[block_id][1]:
                continue
            if self._record_thread_count(env, num_threads, block_id, chosen):
                changed = True
        changed |= self._serialize_nested_auto_tiles(
            fn, block_ids, config, num_threads, budget
        )
        return self._finalize_thread_config(config, num_threads, changed)
