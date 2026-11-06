from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import logging
import operator
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

import torch
from torch._inductor.ops_handler import OpsHandler

from .. import exc
from .ast_extension import expr_from_string
from .emitter_protocols import NO_RETRY

if TYPE_CHECKING:
    import ast
    from collections.abc import Mapping
    import contextlib

    import sympy

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..autotuner.config_priors import ValuePrior
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.llm.profile import LlmPromptProfile
    from ..autotuner.search_surface import BackendSearchSurface
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.settings import DotPrecision
    from ..runtime.settings import PrecompileMode
    from ..runtime.settings import Settings
    from ..runtime.structural_policy import StructuralPolicy
    from .aten_lowering import Lowering
    from .aten_lowering import LoweringContext
    from .autotuner_heuristics.registry import AutotunerHeuristicType
    from .autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from .compile_environment import CompileEnvironment
    from .cute.lane_loops.statements import LanePlacement
    from .cute.lane_loops.statements import LaneScope
    from .device_function import Argument
    from .device_function import DeviceBodyContext
    from .device_function import DeviceFunction
    from .device_ir import DeviceIR
    from .device_ir import ForLoopGraphInfo
    from .device_ir import GraphInfo
    from .device_ir_analysis import DeviceIRAnalysis
    from .emitter_protocols import KernelGenerator
    from .emitter_protocols import RetryPolicy
    from .emitter_protocols import RootEmitter
    from .generate_ast import GenerateAST
    from .helper_function import CodegenInterface
    from .host_function import HostFunction
    from .inductor_lowering import CodegenState
    from .inductor_lowering import GenerateASTFromInductor
    from .program_id import ProgramIDs
    from .reduction_strategy import BlockReductionStrategy
    from .reduction_strategy import ReductionEmitter
    from .reduction_strategy import ReductionStrategy
    from .tile_dispatch import TileStrategyDispatch
    from .tile_strategy import DeviceGridState
    from .tile_strategy import DeviceLoopState
    from .tile_strategy import PerThreadNDTileStrategy
    from .tile_strategy import TileStrategy
    from .tile_strategy import VecLaneWrapper

# Defined at runtime so ``helion.backend_api`` can re-export it.
InductorOpOverrides = OpsHandler[Any]

log: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LauncherInfo:
    """Per-backend info for inlining the dependency-free launcher as a local
    ``helion.runtime`` shim (see :mod:`helion.runtime.precompile`), used by
    ``BoundKernel.to_code(options=OutputCodeOptions(allow_helion_deps=False))``."""

    launcher_module: str  # dotted path of the dep-free launcher, inlined verbatim
    launcher_symbol: str  # public launcher name that module defines
    launcher_alias: str  # underscore alias generated code binds it to
    deps: str  # runtime deps, for the header comment
    # ``helion.runtime.<fn>`` runtime helpers the generated host wrapper calls
    # (besides the launcher); the shim re-exports these so the body runs verbatim.
    runtime_helper_names: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class AutotuneGridPolicy:
    """How a backend limits grid-shaped autotune candidates.

    ``raise_independent_axis_block_size_minimums`` applies the legacy
    search-space heuristic that limits each grid axis separately.
    ``max_programs_per_root_grid`` enables a coupled candidate check on the
    product of all axes in each root grid. Backends may select either policy or
    both explicitly.
    """

    raise_independent_axis_block_size_minimums: bool = True
    max_programs_per_root_grid: int | None = None


def read_launcher_source(module_name: str) -> str:
    """Raw source of a dependency-free launcher module."""
    import importlib
    from pathlib import Path

    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    return Path(module.__file__).read_text()


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """De-duplicate ``items`` keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _validate_subscript_indices(index: list[object]) -> int:
    """Validate supported kernel-tensor indices and count narrowing entries."""
    narrowed = 0
    for value in index:
        if value is None or (
            isinstance(value, slice)
            and (value.start, value.stop, value.step) == (None, None, None)
        ):
            continue
        if isinstance(value, int):
            valid = value >= 0
        elif isinstance(value, slice):
            valid = (
                value.step in (None, 1)
                and isinstance(value.start, int)
                and isinstance(value.stop, int)
                and value.start >= 0
                and value.stop > value.start
            )
        else:
            valid = isinstance(value, torch.SymInt) or (
                isinstance(value, torch.Tensor) and value.ndim == 1
            )
        if not valid:
            raise exc.InvalidIndexingType(repr(value))
        narrowed += 1
    if narrowed > 1:
        raise exc.InvalidIndexingType(repr(index))
    return narrowed


class HandlerOpOverrides:
    """Backend overrides for the shared Inductor op handler.

    ``GenerateASTFromInductor`` asks these before its own lowering of the few
    ops whose backend form needs the handler's lifting facilities
    (``_lift``, ``_to_ast``, ``_create_cast_expr``, ``_expected_tensor_dtype``,
    ``parent_handler``).  Each method returns the lifted result name, or
    ``None`` to run the shared lowering.  The base class overrides nothing.
    """

    def to_dtype(
        self,
        handler: GenerateASTFromInductor,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None,
    ) -> str | None:
        return None

    def rsqrt(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        return None

    def integer_shift(
        self, handler: GenerateASTFromInductor, name: str, x: object, y: object
    ) -> str | None:
        return None

    def neg(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        return None

    def abs(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        return None

    def maximum(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        return None

    def minimum(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        return None


class BackendState(enum.Enum):
    """Maturity of a codegen backend."""

    EXPERIMENTAL = "experimental"  # Warns on use
    BETA = "beta"  # No warning, but no backwards-compat guarantee
    STABLE = "stable"


class DeviceIRStage(enum.Enum):
    """Positions in device-IR lowering at which :meth:`Backend.rewrite_device_ir`
    runs, in pipeline order.

    ``lower_to_device_ir`` calls each stage at most once, in this order (the
    ``REDUCTION_*`` stages from inside
    :meth:`DeviceIR.register_rollable_reductions`, which returns after
    ``REDUCTION_BLOCK_IDS_REGISTERED`` for a reduction-free kernel), each
    between the same core steps a backend's inline block once separated, so
    a backend pass keeps its position relative to ``prepare_graph_lowerings``,
    the reduction roller, the structural analysis and the fact builders.
    """

    # After tracing, before ``rewrite_implicit_random_ops``.
    BEFORE_RANDOM_OPS = "before_random_ops"
    # Before ``prepare_graph_lowerings`` (graphs carry no lowerings yet).
    BEFORE_GRAPH_LOWERINGS = "before_graph_lowerings"
    # After ``prepare_graph_lowerings``.
    AFTER_GRAPH_LOWERINGS = "after_graph_lowerings"
    # After ``validate_host_tensor_usage``, the masking cleanup and the
    # ``range_warp_specialize`` pinning.
    AFTER_WARP_SPECIALIZE_PINNING = "after_warp_specialize_pinning"
    # After the epilogue-subtile candidate detection, before reduction rolling.
    AFTER_EPILOGUE_SUBTILE_DETECTION = "after_epilogue_subtile_detection"
    # ``register_rollable_reductions`` entry: graphs are the traced ones.
    REDUCTION_ROLLING_BEGIN = "reduction_rolling_begin"
    # After the reduction dims are collected, before ``reduction_block_ids``.
    REDUCTION_DIMS_COLLECTED = "reduction_dims_collected"
    # After ``reduction_block_ids``; reduction-free kernels stop here.
    REDUCTION_BLOCK_IDS_REGISTERED = "reduction_block_ids_registered"
    # After the roller pass, before the ``ReductionLoopSpec`` entries.
    REDUCTION_ROLLING_ANALYZED = "reduction_rolling_analyzed"
    # After the ``ReductionLoopSpec`` entries (and the per-loop
    # ``register_reduction_loop_config_slots`` calls).
    REDUCTION_LOOPS_REGISTERED = "reduction_loops_registered"
    # After the multi-root shared-pid ``xyz`` disallow.
    AFTER_MULTI_ROOT_PID_RESTRICTION = "after_multi_root_pid_restriction"
    # After ``DeviceIRAnalysis.build`` and the grid policy (``analysis`` set).
    AFTER_ANALYSIS = "after_analysis"
    # After ``build_kernel_matmul_fact`` (``analysis`` set).
    AFTER_KERNEL_MATMUL_FACT = "after_kernel_matmul_fact"
    # After ``build_pointwise_facts``, the last lowering step (``analysis`` set).
    AFTER_POINTWISE_FACTS = "after_pointwise_facts"


class Backend(abc.ABC):
    """Abstract base class for Helion code generation backends.

    A minimal backend implements the seven abstract members -- ``name``,
    ``dtype_str``, ``acc_type``, ``function_decorator``, ``constexpr_type``,
    ``default_launcher_name`` and ``library_imports`` -- plus whichever of the
    raise-by-default hooks the kernels it compiles reach: the expression
    printers (``program_id_expr``, ``cast_expr``, ``arange_expr``,
    ``grid_index_expr``, ``loop_index_expr``, ``scalar_load_expr``,
    ``where_expr``, ``minimum_expr``, ``zeros_expr``, ``full_expr``,
    ``reshape_expr``, ``broadcast_to_expr``, ``reduction_expr``, the lane /
    thread index and argreduce printers), ``inductor_op_overrides``,
    ``grid_barrier_stmt`` and the dependency-free launcher / JAX export hooks.
    The printers, ``inductor_op_overrides`` and ``grid_barrier_stmt`` raise
    ``BackendUnsupported`` until overridden; the launcher / JAX export hooks
    (``dependency_free_launcher_info``, ``capture_jax_launch_metadata``,
    ``build_jax_fn_code``) raise ``NotImplementedError``.  Every other method
    has a default describing the Triton-style tile backend, so a new backend
    overrides only what differs.  Per-op codegen handlers live in the
    backend package's ``_codegen_modules`` (loaded by
    ``backend_registry.import_backend_codegen``) and the class registers
    through ``backend_registry.register_compiler_backend``.

    The methods are grouped into facets, marked by the section comments below
    in this order: identity and capability predicates; config policy; type
    strings and the expression printer; reduction lowering; reference
    execution; the autotune search surface; indexing, fake tensors and
    block-size constraints; the autotuner (fragments, seeds, benchmarking,
    compile cache); the expression printer for tensors, reductions and casts;
    the generated module skeleton and launch arguments; lowering intercepts
    and the per-thread execution model; the codegen pipeline, backend state
    and registries; host wrapper arguments; runtime hooks; strategy factories
    and the autotune entry point.  A backend-specific decision belongs behind
    a hook in one of these facets with a no-op default.
    ``scripts/check_backend_isolation.py`` fails core code that names the CuTe
    backend (a comparison against its name, a ``BackendUnsupported`` raised
    with its name as a literal, or an import of the CuTe packages) and a
    backend package that imports another backend's.
    """

    # -- identity and capability predicates ------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend name used for codegen dispatch (e.g., 'triton')."""
        ...

    @property
    def state(self) -> BackendState:
        """Maturity of this backend; experimental backends emit a warning."""
        return BackendState.EXPERIMENTAL

    def unsupported(self, detail: str) -> exc.BackendUnsupported:
        """The ``BackendUnsupported`` this backend raises for ``detail``.

        Core raises ``raise env.backend.unsupported("...")`` instead of naming
        a backend in a literal ``BackendUnsupported("<name>", ...)``.
        """
        return exc.BackendUnsupported(self.name, detail)

    def reserved_device_names(self) -> frozenset[str]:
        """Identifiers the device function never allocates as variables.

        The CuTe DSL treats ``_`` as a write-only discard binding, so a host
        local of that name must reach the kernel through a renamed argument.
        """
        return frozenset()

    def output_only_names(self) -> list[str]:
        """Host names of the output-only tensors the launcher returns.

        Filled by ``build_launcher_args`` on backends whose launcher allocates
        the kernel's outputs (Pallas); the host wrapper then assigns the
        launcher's result to these names and allocates their placeholders on
        the meta device.  Empty when the launcher returns nothing.
        """
        return []

    @property
    def lowers_integer_powers(self) -> bool:
        """Whether scalar device expressions prove bounded integer powers
        (``integer_power.prepare_integer_powers``) before printing."""
        return False

    @property
    def requires_proved_integer_powers(self) -> bool:
        """Whether an integer power the proof cannot bound is a refusal
        (``BackendUnsupported``) rather than left to the sympy printer."""
        return False

    @property
    def supports_inter_loop_barriers(self) -> bool:
        """Whether a barrier can be placed between sibling device loops that
        write and then read the same tensor (``needs_barrier_before``)."""
        return False

    def supports_intra_loop_barriers(self, *, in_divergent_control_flow: bool) -> bool:
        """Whether a load that reads a tensor an earlier store of the same
        body wrote can be preceded by a barrier; ``in_divergent_control_flow``
        asks about a load inside a branch whose condition may differ across
        the threads that would synchronize."""
        return False

    @property
    def lowers_chunk_unbind(self) -> bool:
        """Whether ``torch.chunk`` / ``torch.unbind`` of a size-two axis lower
        on the device (through ``hl.split``)."""
        return False

    def barrier_allows_pid_type(self, pid_type: str) -> bool:
        """Whether ``hl.barrier()`` keeps ``pid_type`` available.

        A barrier is a sequential phase boundary between top-level loops; the
        default forces the persistent schedulers.  A backend that launches the
        phases in order itself keeps the pid type it does that under.
        """
        return False

    def local_fetch_add_fake_shape(
        self, target: torch.Tensor, index: list[object]
    ) -> torch.Tensor | None:
        """The fake result of ``hl.atomic_add(..., return_old=True)`` when the
        backend lowers it as a local fetch-add returning one old value per
        logical index; ``None`` for the generic subscript shape."""
        return None

    def transform_launch_grid(self, fn: DeviceFunction, grid_expr: ast.AST) -> ast.AST:
        """Rewrite the host launch grid of ``fn`` (the pid strategy's grid)
        for backend-owned launch structure: extra CTAs per tile, cluster
        dimensions.  Returns ``grid_expr`` unchanged by default."""
        return grid_expr

    @property
    def supports_augmented_assign(self) -> bool:
        """Whether the device language accepts ``x -= y`` on a scalar."""
        return True

    @property
    def folds_static_program_counts(self) -> bool:
        """Whether a statically known program count may be folded into the
        grouped-ordering arithmetic (``L2GroupingProgramIDs``)."""
        return False

    @property
    def reduction_loops_allow_non_power_of_two(self) -> bool:
        """Whether a ``ReductionLoopSpec`` may take a non-power-of-two size."""
        return False

    @property
    def reduction_loops_allow_full_extent(self) -> bool:
        """Whether a ``ReductionLoopSpec`` may take the reduction's full extent."""
        return False

    def canonicalizes_reduction_loop_blocks(self) -> bool:
        """Whether ``CompileEnvironment.canonical_block_id`` follows a
        ``ReductionLoopBlockSizeSource`` to the block it was derived from (a
        backend whose reduction loops share their block's identity)."""
        return False

    @property
    def merges_reduction_dims(self) -> bool:
        """Whether a reduction over several adjacent dims lowers as one
        reduction over a flattened view (``merge_reduction_dims``; needs
        ``static_shapes`` so a later input cannot change the view's strides).
        Off for a backend whose tensor argument handling is untested with
        views."""
        return False

    @property
    def requires_merged_reduction_dims(self) -> bool:
        """Whether a multi-dim reduction chain ``merge_reduction_dims`` cannot
        flatten is a refusal (``BackendUnsupported``) rather than lowered as
        its chain of single-dim reductions."""
        return False

    @property
    def warns_fast_math_unused(self) -> bool:
        """Whether a kernel with fast-math-sensitive operations warns when the
        ``fast_math`` setting is off (``exc.FastMathUnused``)."""
        return False

    @property
    def supports_autotune_multi(self) -> bool:
        """Whether ``Kernel.autotune_multi`` can run on this backend."""
        return False

    @property
    def supports_debug_dtype_asserts(self) -> bool:
        """Whether ``Settings.debug_dtype_asserts`` can run on this backend.

        The setting emits Triton ``tl.cast`` / ``tl.static_assert`` checks
        after every device node, so only backends that generate Triton support
        it.
        """
        return False

    @property
    def supports_eager_prepared_call(self) -> bool:
        """Whether eager calls may reuse a resolved ``BoundKernel`` directly.

        The prepared call still enters the backend's generated host wrapper; it
        only bypasses Helion's repeated binding and specialization lookup.  The
        default is conservative because some backends own call-time behavior in
        :meth:`Kernel.__call__` before the wrapper is reached.
        """
        return False

    @property
    def supports_inductor_fusion(self) -> bool:
        """Whether generated kernels apply Inductor's prologue/epilogue hooks.

        ``torch.compile`` fusion splices Inductor-generated Triton code into the
        kernel through ``load_transform``/``store_transform``.  A backend that
        does not honor those hooks must lower its kernels unfused; otherwise
        Inductor would drop the fused producers and consumers.
        """
        return False

    @property
    def dynamo_captures_kernel_launch(self) -> bool:
        """Whether Dynamo may trace a kernel call inline down to its launch.

        On PyTorch builds without Helion's fusion entrypoints, ``torch.compile``
        traces ``Kernel.__call__`` and the generated host wrapper.  That is how
        Dynamo captures Triton launches (as user-defined Triton kernels) and
        where Pallas's compile capture hooks in, so it stays the default.
        Backends whose launch Dynamo cannot trace return False and are captured
        as the Helion kernel HOP instead.
        """
        return True

    @property
    def dynamo_inline_capture_loses_view_writes(self) -> bool:
        """Whether Dynamo's inline capture drops a launch's write through a view.

        A Triton launch is captured as a user-defined Triton kernel, whose
        functionalization clones a mutated view's base-storage span from the
        materialized view and so loses the write.  On builds with the template
        lowering, a call that writes an input through a view is captured as
        the Helion kernel HOP instead.
        """
        return False

    @property
    def max_tensor_numel(self) -> int | None:
        """Per-tile maximum tensor element count enforced during config search.

        Triton has a hard internal ceiling (currently 2**20) past which its
        codegen rejects the kernel, so the search must avoid generating
        configs that exceed it. Pallas/Mosaic has no analogous compile-time
        cap; tile size is bounded by VMEM bytes (already guarded at runtime
        in :mod:`helion.runtime`). Backends that don't need the cap should
        return ``None`` to disable the constraint.
        """
        from ..autotuner.config_generation import TRITON_MAX_TENSOR_NUMEL

        return TRITON_MAX_TENSOR_NUMEL

    @property
    def pads_small_matmul_dims(self) -> bool:
        """Whether a matmul output tile may be larger than a small M / N
        dimension: the lowering masks the rows / columns past the extent, so
        the autotuner may search output block sizes beyond the dimension
        (``SMALL_DIM_BLOCK_SIZE_OVERSHOOT``), where a hardware-aligned MMA
        tile runs faster than the dimension-sized one.  Backends whose
        lowerings read and write exactly the dimension keep the search
        ceiling at the extent.
        """
        return False

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        """Whether on-device tensor factory ops (zeros/ones/empty/full/...) should
        have their integer dim sizes rounded up to the next power of 2.

        Triton requires power-of-2 block sizes, so the default is True. Pallas
        does not require this and the padding causes broadcast mismatches
        against unpadded full-tensor loads.
        """
        return True

    @property
    def requires_shape_specialized_module(self) -> bool:
        """Whether distinct ``static_shapes`` specializations must compile to
        distinct Python modules (i.e. the generated module holds shape-specific
        mutable state cached across calls).

        The Pallas runtime treats each generated module as the state container for
        one static specialization -- the output-meta descriptor
        (``_helion_output_meta_cache_N``), the launcher cache (``_pallas_cache``),
        the ``_LauncherFastPath`` ds-pad decision, and ``_DirectCallKernel``'s
        signature lock are all monomorphic (populated on first call, reused as-is).
        Since ``PyCodeCache`` keys modules by source text, a shape-polymorphic body
        (e.g. ``compact_worklist``) would otherwise share one module across shapes
        and inherit the first shape's state; the compiler folds the input signature
        into the module cache key to prevent that.

        Backends that allocate outputs fresh each call and hold no shape-dependent
        module state (e.g. Triton) return False and may freely share a module
        across shapes.

        The per-module signature only discriminates shapes that ``bind()`` already
        keys to distinct BoundKernels -- i.e. shapes derived from tensor metadata
        (shape/dtype/stride/device). An output extent driven by an *unbacked* scalar
        arg is collapsed by ``bind()`` to a single BoundKernel, so it is not
        distinguished here; such kernels must ``hl.specialize()`` the scalar to get a
        distinct module per extent.
        """
        return False

    @property
    def codegen_name(self) -> str:
        """Backend name used to look up registered codegen functions."""
        return self.name

    def validate_environment(self) -> None:
        """Raise a ``helion.exc.*`` error if this backend cannot run here.

        Called once per :class:`CompileEnvironment` for the *selected* backend
        (never at registration time), so a backend can hard-require libraries,
        CUDA versions, or hardware and fail fast with an actionable message
        instead of crashing deep in codegen. The default is a no-op.
        """
        return None

    def rng_streams(self) -> tuple[str, ...]:
        """The ``cute_rng_stream`` settings this backend's RNG lowering accepts,
        its default first: a backend without a stream choice keeps the inert
        single-word stream (``word0``)."""
        return ("word0",)

    def collective_owns_tile(self, fn: DeviceFunction, block_id: int) -> bool:
        """Whether a typed physical collective owns an axis's element coordinates."""
        return False

    # -- config policy ---------------------------------------------------------

    def codegen_config(self, config: Config) -> Config:
        """Resolve a backend-owned physical schedule without mutating search input."""
        return config

    def config_value_priors(self, config_spec: ConfigSpec) -> dict[str, ValuePrior]:
        """Per-config-key priors that bias the autotuner's random exploration.

        Returns a mapping from config-key name (e.g. ``"num_warps"``,
        ``"indexing"``, ``"tcgen05_cluster_m"``) to a
        :data:`~helion.autotuner.config_priors.ValuePrior`. Half of the random
        portion of the initial population is drawn using these priors (the other
        half stays uniform), so the search starts denser in the region good
        configs tend to occupy without losing coverage. Keys without a prior --
        and every key when this returns an empty mapping -- are sampled
        uniformly. The default is no bias.
        """
        return {}

    def autotune_config_is_viable(
        self, config_spec: ConfigSpec, config: Config
    ) -> bool:
        """Return whether an automatically generated config is worth compiling.

        This hook only screens candidates produced during autotuning. An explicit
        fixed config still reaches normal backend validation and compilation.
        """
        return True

    def autotune_grid_policy(self, config_spec: ConfigSpec) -> AutotuneGridPolicy:
        """Return how autotuning should limit grid-shaped candidates."""
        return AutotuneGridPolicy()

    # -- type strings and the expression printer (scalars, indices, loops) -----

    @abc.abstractmethod
    def dtype_str(self, dtype: torch.dtype) -> str:
        """Convert a torch dtype to a backend-specific type string.

        For example, Triton returns 'tl.float32' for torch.float32.
        """
        ...

    @abc.abstractmethod
    def acc_type(self, dtype: torch.dtype) -> str:
        """Get the accumulator type string for reductions.

        Some backends may promote certain types for numerical stability
        during reductions (e.g., fp16 -> fp32).
        """
        ...

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        """Get the index type string for the given dtype.

        Defaults to dtype_str, but backends may override for special handling.
        """
        return self.dtype_str(index_dtype)

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        """Expression reading the program id of grid axis ``dim`` as ``index_dtype``."""
        raise exc.BackendUnsupported(self.name, "program IDs")

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        """Ceil-division of ``numel`` by ``block_size`` as an expression;
        ``is_device`` says whether it renders in the kernel or the host wrapper."""
        return f"(({numel}) + ({block_size}) - 1) // ({block_size})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        """Generate a backend-specific type cast expression."""
        raise exc.BackendUnsupported(self.name, "cast")

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        """Render a SymPy expression for this backend's device code."""
        from .triton.printer import texpr

        return texpr(expr)

    @property
    def range_requires_python_int(self) -> bool:
        """Whether range bounds must be plain Python ints (not traced values).

        When True, the codegen will skip dtype casts on range end/step
        expressions so that ``range()`` receives concrete Python integers
        instead of backend-traced values.
        """
        return False

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        """Generate a backend-specific range expression, or None to use the default."""
        return None

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        """Generate a backend-specific arange expression for loop offsets."""
        raise exc.BackendUnsupported(self.name, "arange")

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific grid index expression from an offset."""
        raise exc.BackendUnsupported(self.name, "grid index")

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific device-loop index expression from an offset."""
        raise exc.BackendUnsupported(self.name, "loop index")

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        """Load scalar value from a tensor argument."""
        raise exc.BackendUnsupported(self.name, "scalar load")

    def ast_to_dtype_expr(self, expr_str: str, dtype_str: str) -> str:
        """Generate dtype conversion expression for AST values."""
        return self.cast_expr(expr_str, dtype_str)

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        """Optional per-thread mask restricting active threads to tile width."""
        return None

    # -- reduction lowering ----------------------------------------------------

    def max_reduction_threads(self) -> int | None:
        """Maximum threads for a single warp-level reduction, or None if unlimited."""
        return None

    def max_reduction_loop(self) -> int | None:
        """Maximum user-visible loop chunk for a rolled reduction."""
        return self.max_reduction_threads()

    def default_reduction_loop(self, size_hint: int, max_threads: int) -> int:
        """Loop chunk for a reduction wider than ``max_threads`` that no
        ``reduction_loops`` config entry sizes (``TileStrategyDispatch``).

        Tile-level backends roll it at the thread cap; a thread-level backend
        may cover a wider chunk per trip (more live lanes or per-thread lanes).
        """
        return max_threads

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        """Adjust reduction thread count to fit within hardware thread limits.

        Tile-level backends return the count unchanged. Thread-level backends
        (e.g., CuTe) override this to cap against the per-block thread budget
        shared across all tiled dimensions.
        """
        return requested

    def reduction_block_size_is_inlined_constexpr(self) -> bool:
        """Whether the reduction-loop block size is inlined as a module-level
        literal instead of a constexpr kernel param.

        FlyDSL's scf.for step must be a value produced inside the loop, not an
        external constexpr param, so it inlines the block size as a literal.
        Other backends return False and use a constexpr kernel param.
        """
        return False

    def create_synthetic_reduction_lanes(
        self,
        thread_count: int,
        size_hint: int,
    ) -> int | None:
        """Determine if a synthetic lane loop is needed for a persistent reduction.

        Returns the lane extent when lanes are needed, or None if not.
        Tile-level backends never need lanes. Thread-level backends
        (e.g., CuTe) override this to create lanes when the padded
        reduction size exceeds the thread count.
        """
        return None

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        """Statement emitted between persistent phases, if supported."""
        raise exc.BackendUnsupported(self.name, "hl.barrier()")

    def reduction_axis_first(self) -> bool:
        """Whether reduction strategies should occupy the first (lowest) thread axes."""
        return False

    def supports_lane_loop_reductions(self) -> bool:
        """Whether reductions may be carried by a tile strategy's lane loop."""
        return False

    def validate_reduction_input(self, block_index: int, value: torch.Tensor) -> None:
        """Validate a value before lowering its reduction."""
        return None

    def force_tile_mask(self) -> bool:
        """Whether tile strategies must emit explicit masks for all tiles."""
        return False

    def launches_surplus_tile_threads(self) -> bool:
        """Whether an axis may be launched wider than the tile it indexes."""
        return False

    # -- reference execution ---------------------------------------------------

    def reference_override(
        self, function: object, args: tuple[object, ...]
    ) -> tuple[bool, object]:
        """Optional backend semantic policy for a public operation's reference."""
        return False, None

    def validate_implicit_rng_reference(self) -> None:
        """Validate implicit RNG support under the selected semantic policy."""
        return None

    # -- autotune search surface -----------------------------------------------

    def config_spec_extension(self) -> BackendSearchSurface:
        """The backend's contribution to ``ConfigSpec``'s autotune search surface.

        Stateless: ``ConfigSpec`` stores the object once per spec and calls it
        at the fixed positions of its field table and ``normalize`` pipeline
        (see ``helion/autotuner/search_surface.py``).  The default is the
        shared no-op surface, which adds nothing beyond
        :meth:`tunable_fragments`.
        """
        from ..autotuner.search_surface import EMPTY_SEARCH_SURFACE

        return EMPTY_SEARCH_SURFACE

    def supports_config_key(self, key: str) -> bool:
        """Whether configs for this backend may carry ``key``.

        Two-set rule: a key is accepted when the backend's own search surface
        declares it, or when no backend declares it as specific (so the
        generic keys stay portable across backends).
        """
        from ..autotuner.config_spec import backend_specific_keys

        return (
            self.config_spec_extension().supports_key(key)
            or key not in backend_specific_keys()
        )

    def default_num_sm_multiplier(self) -> int:
        """``num_sm_multiplier`` of a persistent config that names none, and
        the value the autotuner's search starts persistent launches from."""
        from ..autotuner.config_spec import DEFAULT_NUM_SM_MULTIPLIER

        return DEFAULT_NUM_SM_MULTIPLIER

    def cross_loop_pipelines(self) -> tuple[str, ...]:
        """``cross_loop_pipeline`` values this backend can lower.

        Consulted only when the backend supports the ``cross_loop_pipeline``
        config key.
        """
        from ..autotuner.config_spec import VALID_CROSS_LOOP_PIPELINES

        return VALID_CROSS_LOOP_PIPELINES

    # -- indexing, fake tensors and block-size constraints ---------------------

    def supports_block_ptr_indexing(self) -> bool:
        """Whether ``indexing="block_ptr"`` is a valid config value for this backend."""
        return True

    def process_fake_tensor_load(
        self,
        tensor: torch.Tensor,
        index: list[object],
    ) -> None:
        """Called during `type_propagation` when processing a `load` memory op on fake tensors"""
        return

    def normalize_input_fake_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the tensor metadata used while tracing a kernel input."""
        return tensor

    def fake_subscript_shape(
        self,
        tensor: torch.Tensor,
        index: list[object],
    ) -> list[int | torch.SymInt]:
        """Validate a kernel-tensor subscript and return its fake output shape.

        All backends support shape-only ``None`` and full-slice indexing.
        Backends that implement narrowing override this method so unsupported
        indexing cannot reach backend codegen that assumes shape-only views.
        """
        if _validate_subscript_indices(index):
            raise exc.BackendUnsupported(
                self.name, "narrowing kernel-tensor subscripts"
            )

        input_size = collections.deque(tensor.size())
        output_size: list[int | torch.SymInt] = []
        for value in index:
            if value is None:
                output_size.append(1)
            elif isinstance(value, slice) and repr(value) == "slice(None, None, None)":
                output_size.append(input_size.popleft())
            else:
                raise exc.InvalidIndexingType(repr(value))
        assert len(input_size) == 0
        return output_size

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        """Adjust block-size min/max constraints for backend-specific alignment.

        Called after all block-size specs have been created.  ``block_specs``
        is a list of ``BlockSizeSpec`` objects (one per tiled dimension).
        ``ndim`` is the total number of tiled dimensions.
        ``block_sizes``, ``kernel_tensor_sizes``, and ``min_element_bits``
        provide additional context for backends that need physical tensor
        dimension info.

        The default does nothing.  Backends with alignment requirements
        (e.g., Pallas/TPU) override this to enforce minimums.
        """
        return

    # -- autotuner: fragments, seeds, benchmarking, compile cache --------------

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Extra ``ConfigSpec`` fragments (config key -> fragment) this backend
        adds to every kernel's search space; ``{}`` for none."""
        return {}

    def autotuner_heuristics(self) -> tuple[AutotunerHeuristicType, ...]:
        """The compiler-seed heuristics this backend plants, in ranked order.

        The order is part of a seed list's identity (prefix contracts, the
        promoted default and the autotune cache fingerprint depend on it).
        ``()`` means the backend plants no compiler seeds.
        """
        return ()

    def seed_fact_heuristics(self) -> tuple[AutotunerHeuristicType, ...]:
        """Heuristics outside :meth:`autotuner_heuristics` that own cache facts.

        :meth:`postprocess_seed_configs` may append seeds from heuristics kept
        out of the ranked tuple on purpose; naming them here lets a fired one
        still contribute its ``CACHE_SPECIALIZATION_FACTS``.
        """
        return ()

    def register_seed_facts(
        self, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        """Register backend-wide correctness facts before any seed is built.

        Runs next to every heuristic's ``register_facts``, also when autotuner
        heuristics are disabled. Returns the runtime facts the registration
        depends on (see ``AutotunerHeuristic.register_facts``).
        """
        return frozenset()

    def postprocess_seed_configs(
        self, env: CompileEnvironment, device_ir: DeviceIR, configs: list[Config]
    ) -> list[Config]:
        """Rewrite the deduplicated compiler seed list after every heuristic ran.

        Backends append carriers and derived seeds here; the order they return
        is kept (only exact duplicates are folded afterwards).
        """
        return configs

    def register_coverage_groups(
        self, env: CompileEnvironment, device_ir: DeviceIR
    ) -> None:
        """Declare compiler coverage groups once every ordinary field is final."""
        return

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]] | None:
        """Return the benchmarking function for this backend.

        The default returns ``None`` which causes the autotuner to use the
        module-level ``do_bench`` (patchable by tests).  Backends that need
        a different timing mechanism (e.g., Pallas/TPU) should override
        this to return their own function.
        """
        return None

    def prefers_wall_clock_bench(self) -> bool:
        """Whether the subprocess benchmark worker times with the wall clock
        when this backend's ``get_do_bench`` returns the generic
        ``do_bench_generic`` (``benchmark_provider._subprocess_benchmark_uses_wall_clock``).
        A backend that keeps the default ``get_do_bench`` (``None``) is timed
        with device events and is never asked."""
        return False

    def llm_prompt_profile(self) -> LlmPromptProfile:
        """The backend-specific prose of the LLM autotuner's prompts
        (``helion.autotuner.llm``): the default names the backend and
        describes the Triton-style config space."""
        from ..autotuner.llm.profile import LlmPromptProfile

        return LlmPromptProfile(self.name)

    def final_rebenchmark_top_k(self) -> int | None:
        """How many finalists the autotuner's final rebenchmark verifies, or
        ``None`` for the autotuner's default (``base_search``)."""
        return None

    def final_rebenchmark_isolated(self) -> bool:
        """Whether the final rebenchmark times each finalist in its own steady
        window by default rather than interleaved with its rivals."""
        return False

    def probe_long_autotune_kernels(self, config_spec: ConfigSpec) -> bool:
        """Whether candidate timing should first probe for a long-running kernel.

        The probe avoids repeatedly executing a candidate whose first measured
        call already exceeds the benchmark's warmup and measurement windows.
        """
        return False

    def get_interleaved_bench(
        self,
    ) -> Callable[..., list[float]] | None:
        """Return the interleaved benchmarking function for this backend.

        The default returns ``None`` which causes the autotuner to use the
        module-level ``interleaved_bench``.  Backends without Triton event
        timing should override.
        """
        return None

    @classmethod
    def finalist_preference(cls, config: Config, source: Callable[[], str]) -> int:
        """Rank of ``config`` among the autotuner's finalists it cannot tell
        apart (``BaseSearch._prefer_backend_preference_within_tolerance``):
        lower wins, equal ranks keep the faster config.  For what a config
        costs or saves outside the kernel's own isolated timing; the default
        prefers nothing.  ``source`` renders the finalist's generated module
        on demand (the rendered kernel, not the config's keys, says which
        lowering a config took).  A class method: the search reaches it
        through the backend's registered name (``Settings.backend``), without
        an instance.
        """
        return 0

    def get_paired_device_micros_bench(
        self,
    ) -> Callable[..., list[tuple[float, float]]] | None:
        """Paired device-µs bench for the autotune final-pick re-rank, or None.

        Backends that can cheaply report per-call on-device µs override this to
        return a callable ``fn(candidates, reference, *, desc) ->
        list[(candidate_device_micros, paired_delta_micros)]``. The default returns None,
        leaving final-pick on its wall-clock rebench.
        """
        return None

    def supports_precompile(self) -> bool:
        """Whether this backend supports subprocess precompilation.

        Triton backends use fork/spawn to precompile kernels and detect hangs.
        Other backends (Pallas, CuTe) may not need or support this.
        """
        return True

    def supports_parallel_render(self) -> bool:
        """Whether autotune candidates may be rendered in forked workers.

        Code generation for such a backend is a pure function of the bound
        kernel and the config whose only CUDA reads are the device facts
        ``helion.autotuner.render_pool.CudaDeviceFacts`` captures, so a child
        forked from the autotuning process renders byte-identical source
        without touching the driver. Backends with a precompile pool do not
        need it; see :mod:`helion.autotuner.render_pool`.
        """
        return False

    def setup_compile_cache_dir(self, device_index: int) -> None:
        """Point the backend's on-disk compile cache at Helion's cache root.

        Called from :meth:`BoundKernel.compile_config` before compilation.
        Backends that use a per-device on-disk cache of compiled artifacts
        (Triton, CuTe) override this to set the relevant environment variable
        (respecting any user override).  The default is a no-op.
        """
        return None

    def make_ephemeral_cache(
        self,
    ) -> contextlib.AbstractContextManager[None] | None:
        """Return a context manager that redirects the on-disk compile cache
        to a throwaway directory during autotuning, or ``None`` when the
        backend has no ephemeral-cache behavior.

        Autotuning compiles many candidate configs; without this they would
        pollute the persistent cache.  The winning config's artifact is
        restored into the real cache afterward (see
        :meth:`finalize_ephemeral_cache`).
        """
        return None

    @staticmethod
    def keep_compile_cache_requested() -> bool:
        """Whether the user asked to keep every candidate's compile-cache
        artifact during autotuning (i.e. disable the ephemeral cache).

        ``HELION_KEEP_CACHE`` is the backend-agnostic control, matching the
        rest of the ``HELION_*CACHE*`` env-var family.
        """
        return os.environ.get("HELION_KEEP_CACHE", "") == "1"

    def finalize_ephemeral_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        """Post-autotune cleanup after running inside an ephemeral cache.

        Restores the winning config's artifact into the real (persistent)
        cache: CuTe re-persists the in-memory compiled module directly;
        Triton evicts the in-memory artifact so the next call recompiles
        into the real cache.  No-op by default.
        """
        return None

    def compiled_cache_key(
        self, bound_kernel: BoundKernel[Any], compiled_fn: object
    ) -> str | None:
        """Return a stable backend cache key for an already-compiled callable.

        ``compiled_fn`` is the value stored in ``bound_kernel._compile_cache``
        for the requested config.  Returns ``None`` if the backend has no cache
        key or the kernel has not been JIT-compiled yet.
        """
        return None

    def annotate_compiled_module(
        self, module: object, source: str, kernel_name: str
    ) -> None:
        """Record codegen metadata on a freshly-loaded generated module.

        Called from :meth:`BoundKernel.compile_config` after the generated
        source has been imported.  Backends that derive a cross-process compile
        cache key from the generated source (CuTe) override this.  No-op default.
        """
        return None

    def generated_source_hash(self, compiled_fn: object) -> str | None:
        """Return the exact generated-source hash attached to ``compiled_fn``.

        Backends may return ``None`` when generated-source identity is not
        available. The autotuner uses a non-``None`` value only for
        backend-scoped effective-code deduplication.
        """
        return None

    def should_deduplicate_generated_sources(self, config_spec: ConfigSpec) -> bool:
        """Whether the autotuner may merge source-identical candidates.

        This is disabled by default because source identity is not necessarily
        sufficient to establish benchmark equivalence for every backend and
        search domain.
        """
        return False

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        """Classify an exception that occurred during autotuning.

        Returns one of:
          - ``"raise"``: unexpected error, caller should re-raise
          - ``"warn"``:  notable but expected; log as warning
          - ``"debug"``: benign/expected; log at debug level
          - ``None``:    backend has no opinion; fall through to default

        The default returns ``None`` so the existing Triton-oriented
        classifier handles it.
        """
        return None

    # -- expression printer: tensors, reductions and casts ---------------------

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        """Generate a backend-specific conditional select expression."""
        raise exc.BackendUnsupported(self.name, "where")

    def minimum_expr(self, a: str, b: str) -> str:
        """Generate a backend-specific minimum expression."""
        raise exc.BackendUnsupported(self.name, "minimum")

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        """Generate a backend-specific arange expression for reduction index setup."""
        raise exc.BackendUnsupported(self.name, "arange index")

    def zeros_expr(self, shape: str, dtype: str) -> str:
        """Generate a backend-specific zeros expression."""
        raise exc.BackendUnsupported(self.name, "zeros")

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        """Expression of a ``shape_dims`` tensor filled with ``value_expr`` in ``dtype``."""
        raise exc.BackendUnsupported(self.name, "full tensor creation")

    def reduction_acc_init_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        """Initial value of a rolled reduction's per-thread accumulator.

        Separate from :meth:`full_expr` because the accumulator must be as wide
        as the backend's combine expression, which may promote (Metal reduces
        ``int8``/``bool`` in an ``int``).  Declaring it at storage width would
        truncate on every loop iteration.
        """
        return self.full_expr(shape_dims, value_expr, dtype)

    def looped_reduction_thread_count(
        self,
        *,
        requested: int,
        block_size: int,
        block_index: int,
        config: Config,
        config_spec: ConfigSpec,
    ) -> int | None:
        """Backend override for the per-block thread count of a looped whole-row
        reduction.

        Return None to keep the default (``requested``). Tile-level backends
        return None.
        """
        return None

    def register_reduction_loop_config_slots(
        self,
        env: CompileEnvironment,
        block_id: int,
        size_hint: int,
    ) -> None:
        """Register backend-specific config-spec slots for a rollable reduction dim.

        Called once per rollable rdim during device_ir analysis, after the shared
        ``ReductionLoopSpec`` has been appended. Default: no-op. Backends that need
        additional tuning knobs per reduction block (e.g. flydsl's
        ``cute_vector_widths``) override this instead of adding a name-check in
        device_ir.
        """
        return

    def wrap_reduction_accumulator(
        self,
        acc_full: str,
        *,
        thread_count: int,
        loop_block_size: int,
        acc_dtype: torch.dtype,
    ) -> str:
        """Wrap the looped-reduction accumulator init expression, if needed.

        FlyDSL's runtime scf.for carries the accumulator as an iter_arg whose init
        type must match the per-thread vector the loop body yields, so it wraps
        the scalar seed in ``fx.Vector.filled(...)``. Identity by default.
        """
        return acc_full

    def reshape_expr(self, expr: str, shape: str) -> str:
        """Expression reshaping ``expr`` to ``shape``."""
        raise exc.BackendUnsupported(self.name, "reshape")

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        """Expression broadcasting ``expr`` to ``shape``."""
        raise exc.BackendUnsupported(self.name, "broadcast_to")

    def maybe_reshape_reduction(
        self,
        expr: str,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        target_shape_expr: str,
    ) -> str:
        """Reshape a reduction result from its physical to logical shape."""
        return self.reshape_expr(expr, target_shape_expr)

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        """Generate the index expression for a reduction dimension."""
        raise exc.BackendUnsupported(self.name, "reduction index")

    def reduction_index_zero_expr(self, dtype: str) -> str:
        """Generate the zero-length index expression for an empty reduction."""
        raise exc.BackendUnsupported(self.name, "reduction index zero")

    def next_power_of_2_host_expr(self, expr: str) -> str:
        """Generate a host-side next-power-of-2 expression."""
        raise exc.BackendUnsupported(self.name, "next_power_of_2")

    def static_rdim_size(self, numel: int) -> int:
        """Return the RDIM block size for a statically known reduction dimension."""
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        return next_power_of_2(numel)

    def dynamic_rdim_size_expr(self, expr: str) -> str:
        """Generate a host-side expression for RDIM size from a dynamic dimension.

        By default delegates to next_power_of_2_host_expr. Backends like Pallas
        that need exact sizes can override to return the expression unchanged.
        """
        return self.next_power_of_2_host_expr(expr)

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        """Thread index expression with elements-per-thread stride for lane loops."""
        raise exc.BackendUnsupported(self.name, "lane index")

    def lane_offset_expr(self, lane_var: str) -> str:
        """Cast a lane variable for addition to an index expression."""
        raise exc.BackendUnsupported(self.name, "lane offset")

    def thread_index_expr(self, *, axis: int) -> str:
        """Bare thread index expression (no elements-per-thread stride),
        used by the strided lane layout."""
        raise exc.BackendUnsupported(self.name, "thread index")

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        """Generate the combine expression for looped reductions."""
        from torch._inductor.ir import get_reduction_combine_fn

        combine_fn = get_reduction_combine_fn(reduction_type, dtype)
        return str(combine_fn(acc, val))

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
        """Generate the cross-thread reduction expression.

        ``dtype`` is the accumulation dtype (``reduction_acc_dtype``: the
        output's computation dtype, the input's for argmin/argmax) when the
        caller knows it.
        Backends that allocate typed scratch storage for the reduction (e.g.
        Metal's ``threadgroup`` buffers) need it; the rest ignore it.
        """
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def thread_linear_index_expr(self, axis_sizes: dict[int, int]) -> str | None:
        """Linearized thread index expression for active block axes, if available."""
        return None

    def reduction_threads_hint(self, block_size_var: str | None = None) -> int | None:
        """Best-effort thread count used by reduction_expr for the given block size."""
        return None

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        """Whether this reduction type tracks an auxiliary index state."""
        return False

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
        """Generate the cross-thread argmin/argmax expression.

        ``dtype`` is the accumulation dtype of the *value* operand; see
        :meth:`reduction_expr`.
        """
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

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
        """Per-iteration accumulator update for a rolled argmin/argmax.

        ``dtype`` is the accumulation dtype of the value operand; see
        :meth:`reduction_expr`.
        """
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

    def inductor_op_overrides(self) -> InductorOpOverrides:
        """The Inductor ``OpsHandler`` the pointwise lowering prints elementwise ops through."""
        raise exc.BackendUnsupported(self.name, "Inductor OpOverrides")

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        """``x`` cast to ``target_dtype`` as an AST (``cast_expr`` over ``dtype_str``)."""
        return expr_from_string(
            self.cast_expr("{x}", self.dtype_str(target_dtype)),
            x=x,
        )

    def join_cast_template(self) -> str | None:
        """An ``expr_from_string`` template casting ``{x}`` to ``{like}``'s type.

        A dynamic if whose branch reassigns a variable joins the new value
        with the old one, which must have one type.  Backends whose emitted
        values can carry a type other than their FX dtype return a cast here;
        ``None`` emits no join (CuTe joins in its own if codegen).
        """
        return None

    def scalar_float_operand_ast(
        self, x: ast.AST, value: torch.SymFloat, *, computes_in_float64: bool
    ) -> ast.AST:
        """``x`` computes ``value``, a SymFloat operand (a Python-float host
        scalar) of an Inductor op.

        The op computes with it as a float32 scalar, as torch rounds such a
        scalar to float32 (also the opmath of fp16/bf16 math, see
        ``opmath_scalar_dtype``) unless the op's math is float64.  Backends
        whose emitted scalar can carry another type cast it here.
        """
        return x

    def cast_scalar_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        """Cast a plain scalar (e.g. a bare number lifted from an index expr) to
        ``target_dtype``.

        Defaults to ``cast_ast``. Backends that write casts as ``value.to(dtype)``
        must override this, because a bare number has no ``.to()`` method --
        FlyDSL, for example, uses ``fx.Float16(5)`` instead.
        """
        return self.cast_ast(x, target_dtype)

    def expands_broadcast_dims(self) -> bool:
        """Whether the backend needs Triton-style ``[None, :]`` broadcast-expand
        of sub-rank tensors.

        Tile-level backends (Triton, etc.) broadcast-expand a sub-rank operand up
        to the output rank. Backends whose per-thread vectors carry the tile/row
        axis implicitly (e.g. FlyDSL) return False to skip the expansion.
        """
        return True

    # -- generated module skeleton, launcher name and launch arguments ---------

    @property
    @abc.abstractmethod
    def function_decorator(self) -> str:
        """Expression string for the kernel function decorator.

        For example, Triton returns 'triton.jit'.
        """
        ...

    def function_decorator_for_args(self, args: Sequence[Argument]) -> str:
        """Expression string for the kernel function decorator.

        Backends can override this when the decorator needs to depend on the
        generated function signature.
        """
        return self.function_decorator

    @property
    @abc.abstractmethod
    def constexpr_type(self) -> str:
        """Type annotation string for compile-time constant arguments.

        For example, Triton returns 'tl.constexpr'.
        """
        ...

    def inline_constexpr(self, name: str, value: str) -> str:
        """Return the source for a module-level inlined constexpr assignment.

        For example, Triton returns '_BLOCK_SIZE_0 = tl.constexpr(256)'.
        """
        return f"{name} = {self.constexpr_type}({value})"

    @property
    @abc.abstractmethod
    def default_launcher_name(self) -> str:
        """Name of the default host-side launcher symbol for this backend."""
        ...

    def get_launcher_name(self) -> str:
        """Return the launcher name to use for the current config.

        Subclasses can override to select a different launcher based on
        the active configuration (e.g., pipeline launcher).
        """
        return self.default_launcher_name

    @property
    @abc.abstractmethod
    def library_imports(self) -> dict[str, str]:
        """Mapping of short names to import statements for generated code.

        Keys are the short names used in generated code (e.g., 'tl'),
        values are the corresponding import statements.
        """
        ...

    def embedded_helper_source(self, body: str) -> str:
        """Source of in-kernel runtime helpers to inline into the generated module.

        Backends that call a helion-defined helper from inside the generated
        kernel can return its source here (instead of importing it) so the output
        is self-contained -- which lets the precompiler produce a helion-free
        standalone. Only helpers actually referenced in ``body`` should be
        emitted. Injected between the imports and the kernel body. Default: none.
        """
        return ""

    @property
    def dependency_free_launcher_info(self) -> LauncherInfo:
        """Info for inlining this backend's dependency-free launcher as a local
        ``helion.runtime`` shim, used by
        ``to_code(options=OutputCodeOptions(allow_helion_deps=False))``. Backends
        that cannot yet emit a helion-free module raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"the {self.name!r} backend does not support "
            "to_code(allow_helion_deps=False) yet"
        )

    def build_standalone_code(
        self,
        bound: BoundKernel[Any],
        import_lines: list[str],
        body_root: ast.Module,
    ) -> ast.Module | None:
        """Optionally specialize standalone launch code for the bound workload."""
        return None

    def capture_jax_launch_metadata(
        self, bound: BoundKernel[Any], config: Config | dict[str, object]
    ) -> object:
        """Capture jax_fn launch metadata (Pallas only) by compiling the kernel and
        running a capturing launch on real tensors -- must run *outside* the
        fake-tensor env. Consumed by :meth:`build_jax_fn_code`; backends without a
        JAX launch path raise."""
        raise NotImplementedError(
            f"the {self.name!r} backend does not support to_code(jax_fn=True)"
        )

    def build_jax_fn_code(
        self,
        body_root: ast.Module,
        import_lines: list[str],
        meta: object,
        *,
        allow_helion_deps: bool,
    ) -> ast.Module:
        """Rewrite the generated module AST into a jax-native standalone module
        (Pallas only). The entrypoint operates on ``jax.Array`` inputs;
        ``allow_helion_deps`` toggles whether the launch core is inlined (helion-free)
        or imported from helion. ``meta`` is the value from
        :meth:`capture_jax_launch_metadata`. Backends without a JAX launch path raise.
        """
        raise NotImplementedError(
            f"the {self.name!r} backend does not support to_code(jax_fn=True)"
        )

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        """Keyword arguments appended to the generated launch call
        (``num_warps=...`` and friends); ``has_barrier`` says whether the kernel
        contains a grid barrier."""
        return []

    def effective_num_warps(self, config: Config) -> int:
        """Return the warp count the backend will actually launch."""
        return config.num_warps

    # -- lowering intercepts and the per-thread execution model ----------------

    def customize_ast(self, hf: HostFunction) -> bool:
        """Run backend-specific AST customizations.

        Called after static loop unrolling but before type propagation
        and tracing.  Backends can override this to rewrite the user's
        AST for algorithmic transformations that change loop structure.
        Returns whether the AST was rewritten.
        """
        return False

    def pre_inductor_lowering(self, node: torch.fx.Node) -> Lowering | None:
        """Return a backend-owned lowering that must bypass Inductor IR.

        Most ATen operations use Helion's shared Inductor lowering. Backends may
        override this for operations whose Inductor representation cannot be
        consumed by that shared path.
        """
        return None

    def argreduce_uses_aten_lowering(self, node: torch.fx.Node) -> bool:
        """Whether this ``argmax``/``argmin`` node takes the registered ATen
        lowering instead of Inductor's reduction lowering."""
        return False

    def emit_virtual_clone(
        self, ctx: LoweringContext, node: torch.fx.Node
    ) -> object | None:
        """Codegen for an ``aten.clone`` the backend elides; ``None`` runs the
        shared pointwise lowering."""
        return None

    def emit_completed_matmul_sum(
        self, ctx: LoweringContext, node: torch.fx.Node
    ) -> ast.AST | None:
        """Codegen for a reduction the backend already completed inside a
        matmul; ``None`` runs the shared reduction lowering."""
        return None

    def reduction_active_block_id(
        self, state: CodegenState, node: torch.fx.Node, size: object
    ) -> int | None:
        """The active tile block a non-reduction-dim reduction over ``size``
        reduces across (a block-level reduction strategy); ``None`` keeps the
        node's own block."""
        return None

    def handler_op_overrides(self) -> HandlerOpOverrides:
        """Overrides for the shared Inductor op handler's lifting methods."""
        return HandlerOpOverrides()

    def deferred_node_codegen(
        self,
        cg: CodegenInterface,
        node: torch.fx.Node,
        operand_env: Mapping[torch.fx.Node, object],
    ) -> object | None:
        """A stand-in result for a node whose codegen a later emitter owns;
        ``None`` lowers the node now."""
        return None

    def exchanged_node_operands(
        self,
        cg: CodegenInterface,
        node: torch.fx.Node,
        operand_env: Mapping[torch.fx.Node, object],
    ) -> dict[torch.fx.Node, ast.AST]:
        """Operand ASTs substituted while lowering ``node`` only (values other
        threads own); empty for a plain lowering."""
        return {}

    def prepare_graph_codegen(self, cg: GenerateAST, graph: torch.fx.Graph) -> None:
        """Run before a graph's nodes are lowered (``codegen_call_with_graph``)."""
        return None

    def tracks_statement_owners(self) -> bool:
        """Whether ``GenerateAST`` records, per owning fx node, the statements
        a lowering emitted (``_statements_by_owner_node_id``): the bookkeeping
        a backend's root rewrites consult.  ``False`` by default."""
        return False

    def tile_begin_expr(self, codegen: GenerateAST, block_idx: int) -> str | None:
        """The uniform first index of the current tile along ``block_idx``
        (``tile.begin`` / ``tile.end`` / ``tile.id`` render through it), or
        ``None`` to take the owning strategy's ``tile_begin_var``.  A backend
        whose thread / lane partition defines the tile's begin differently
        overrides this."""
        return None

    def rewrite_lowered_root(
        self, codegen: GenerateAST, grid_state: DeviceGridState, body: list[ast.AST]
    ) -> bool:
        """Replace the ordinarily lowered body of a root grid loop in place,
        after ``codegen_call_with_graph`` rendered it and before the grid
        state wraps it in its lane loops; ``True`` when the body was replaced
        (``GenerateAST.visit_For`` then emits it between the grid state's
        prefix and suffix without the lane-loop wrap).  Also the point where a
        backend records facts about the rendered root (the launch axes it
        reads).  ``False`` by default."""
        return False

    def explain_root_failure(
        self, codegen: GenerateAST, error: exc.BackendUnsupported
    ) -> exc.BackendUnsupported | exc.InvalidConfig:
        """The exception to raise for ``error``, a refusal of the ordinary
        lowering or of the device-function finalization: a backend that knows
        a prior owner declined the root for a config-level reason reports that
        reason instead.  Returns ``error`` by default."""
        return error

    def tile_loop_paths(
        self, fn: DeviceFunction
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Maximal stacks of simultaneously active non-reduction tiles (block-id
        tuples, outermost first) whose sibling paths reuse thread axes; empty
        when every tile loop claims its own axes."""
        return ()

    def constexpr_live_graph_ids(self, fn: DeviceFunction) -> frozenset[int] | None:
        """Graph ids reachable without entering the untaken side of a branch
        whose test is a constant for this config; ``None`` when no branch test
        is such a constant."""
        return None

    def constexpr_live_block_ids(
        self, fn: DeviceFunction, live_graph_ids: frozenset[int]
    ) -> frozenset[int]:
        """Block ids the graphs in ``live_graph_ids`` still reference.  Only
        asked when :meth:`constexpr_live_graph_ids` returned graphs, which the
        base backend never does."""
        return frozenset()

    def shares_sibling_thread_axes(self) -> bool:
        """Whether sibling tile loops and mutually exclusive pid branches share
        CUDA thread axes: the dispatcher then widens narrower siblings to the
        launch and lets exclusive branches share an axis."""
        return False

    def inactive_broadcast_alias_block_ids(
        self, fn: DeviceFunction, strategies: Sequence[TileStrategy]
    ) -> set[int]:
        """Reduction blocks that merely alias an existing tile's coordinates
        (register broadcasting) and get no reduction strategy of their own."""
        return set()

    def roots_may_reserve_distinct_thread_axes(self) -> bool:
        """Whether separate root loops may reserve different synthetic or
        reduction axes, so a grid mask is elided only if every root's axes
        fit the extent."""
        return False

    def pointwise_op_is_pure_scalar(self, node: torch.fx.Node) -> bool:
        """Whether a node from the pointwise fact's unsafe-op set (inline
        assembly) is a pure per-element scalar computation the fact may still
        describe."""
        return False

    def rolls_free_iota_reductions(self) -> bool:
        """Whether a reduction fed by a free rank-one ``hl.arange`` (one that
        indexes no load) may still roll into a reduction loop."""
        return False

    def free_iota_reductions(
        self, graphs: list[GraphInfo]
    ) -> Mapping[torch.fx.Node, tuple[torch.fx.Node, ...]]:
        """The free-iota reductions a backend lowering takes over; only those
        roll (:meth:`rolls_free_iota_reductions`)."""
        return {}

    # Policy of the shared per-thread tile planner
    # (``per_thread_planning.plan_per_thread_loop_strategy``, used by CuTe and
    # Metal).

    def boundary_only_grid_block_ids(self, fn: DeviceFunction) -> set[int]:
        """1-D grid tiles whose elements the body never observes (they only
        bound an inner tile), so the per-thread planner gives them no threads."""
        return set()

    def tile_loop_thread_count(
        self, fn: DeviceFunction, config: Config, inactive_grid_block_ids: set[int]
    ) -> int:
        """Threads the simultaneously active grid and device-loop tiles of
        ``fn`` claim under ``config``, for the per-thread planner's budget.

        The default multiplies every active non-reduction loop block's thread
        count (Metal's allocation policy); CuTe budgets sibling passes that
        reuse axes instead.
        """
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        total_threads = 1
        for block_id in _active_loop_block_ids(fn):
            info = env.block_sizes[block_id]
            if info.reduction:
                continue
            bs = info.from_config(config)
            if isinstance(bs, int):
                nt = int(
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                )
                total_threads *= nt if nt > 0 else bs
        return total_threads

    def packed_grid_block(self, config: Config) -> tuple[int, int] | None:
        """``(block_id, tiles_per_block)`` of the grid axis packed several
        tiles wide under ``config`` (its thread axis holds one tile per
        sub-group of threads), or ``None`` when no axis is packed."""
        return None

    def packed_grid_block_id(self, config: Config) -> int | None:
        """Block id of the grid axis packed several tiles wide, which stays a
        thread axis even when the body never distributes over it."""
        packed = self.packed_grid_block(config)
        return None if packed is None else packed[0]

    def matmul_contraction_thread_reserve(
        self, fn: DeviceFunction, tile_block_ids: list[int]
    ) -> int:
        """Threads the per-thread planner reserves for the matmul-contraction
        reduction axes outside ``tile_block_ids`` before it splits what is left
        over the free tile axes (``plan_per_thread_loop_strategy``).  The
        default reserves each such axis's full power-of-two extent, capped at
        ``max_reduction_threads``, for a scalar contraction that cannot walk K
        in lanes; a backend whose matmul lowering contracts its own K returns
        1 (Metal)."""
        from . import per_thread_planning

        return per_thread_planning.matmul_contraction_thread_reserve(
            self, tile_block_ids
        )

    def free_auto_thread_axis_count(self, fn: DeviceFunction, config: Config) -> int:
        """The kernel's free (non-reduction) tile axes that auto-thread and so
        share the budget the contraction reserve leaves; the planner splits
        that budget so their product stays within it."""
        from . import per_thread_planning

        return per_thread_planning.free_auto_thread_axis_count(fn, config)

    def per_thread_mma_candidate_loop(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        *,
        block_sizes: Sequence[int | torch.SymInt],
        num_threads_config: Sequence[int],
        grid_ids: set[int],
    ) -> bool:
        """Whether the device loop over ``block_ids`` is a tensor-core
        contraction candidate: the per-thread planner then keeps its axes
        active, its configured thread counts and the full thread budget.

        The default is the backend-neutral operand test
        (``per_thread_planning.matmul_loop_contracts_root_grid``): an
        accumulating fp32 / fp16 / bf16 matmul whose output axes are the
        root grid's and whose contraction axis is in the loop.  A backend
        with its own operand analysis answers from it instead.
        """
        from .per_thread_planning import matmul_loop_contracts_root_grid

        return matmul_loop_contracts_root_grid(
            fn,
            block_ids,
            block_sizes=block_sizes,
            num_threads_config=num_threads_config,
            grid_ids=grid_ids,
        )

    def per_thread_matmul_loop_has_mma_path(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> bool:
        """Whether a multi-dimensional device loop containing a matmul has a
        tensor-core lowering under ``config``; without one the per-thread
        planner caps its automatic thread counts at 256.  The default answers
        yes when any device loop of the kernel is a contraction candidate
        (``per_thread_mma_candidate_loop``), as the CuTe planner's probe did
        for every backend before the planner moved to core: Metal's root
        tiles beside a tensor-core-dtype K loop keep the full budget."""
        from . import per_thread_planning

        return per_thread_planning.kernel_has_contraction_candidate_loop(
            self, fn, config
        )

    def per_thread_mma_mode(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        config: Config,
        *,
        block_sizes: Sequence[int | torch.SymInt],
    ) -> bool:
        """Whether a dedicated tensor-core body (MMA, flash, warp-MMA) emits
        the device loop over ``block_ids``, so the per-thread strategy
        suppresses its synthetic lane loops (``mma_mode``)."""
        return False

    def per_thread_root_thread_counts(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        config: Config,
        *,
        block_sizes: Sequence[int | torch.SymInt],
        num_threads_config: Sequence[int],
        original_num_threads_config: Sequence[int],
    ) -> list[int] | None:
        """Per-axis thread counts of the root tile ``block_ids`` when a
        tensor-core root plan fixes them (``num_threads_config`` is the
        planner's resolution so far, ``original_num_threads_config`` the
        config's), or None to keep the planner's."""
        return None

    # Thread layout and lane loops of the tile strategies (``tile_strategy``).
    # The defaults describe a backend without a per-thread layout; the
    # per-thread strategies are shared by CuTe and Metal, so each CuTe
    # behaviour they branch on is a hook here.

    def loop_var_prefixes(self) -> tuple[str, str]:
        """Prefixes of the tile-offset variables (``offset_<block>``) and the
        lifted loop-step variables (``step_<n>``) the tile strategies name.
        A backend whose DSL reserves those names answers others."""
        return ("offset", "step")

    def plans_launch_thread_axes(self) -> bool:
        """Whether the launch's thread layout is planned kernel-wide from the
        strategies' axes (``TileStrategyDispatch.thread_axis_for_strategy``):
        the body's thread axes are then checked against the plan, a
        barrier-phase kernel takes the dispatcher's reservation, and an
        argument-sized tile records the axis it lives on
        (``ThreadAxisTracker.record_symbolic_axis``)."""
        return False

    def single_thread_tile_claims_axis(self) -> bool:
        """Whether a flattened tile whose thread extent is one still claims
        its thread axis (``offsets = pid * BLOCK + thread_idx[axis]`` and the
        ``thread_idx[axis] < BLOCK`` mask term).  ``False`` leaves the axis
        free for a synthetic thread axis to reuse."""
        return True

    def flat_strided_tile_walks_elements(self) -> bool:
        """Whether a single-block flattened tile over a strided range
        (``hl.tile(begin, end, step)``) on a thread axis walks the elements of
        the range (``numel = end - begin``, ``index = begin + offset``) rather
        than its trips (``index = begin + offset * step``)."""
        return False

    def tracks_flat_grid_origins(self) -> bool:
        """Whether ``FlattenedTileStrategy`` records each block's uniform tile
        origin (``grid_origin_var``) for the backend's root emitters."""
        return False

    def reflattens_pointwise_tiles(self) -> bool:
        """Whether a flatten-loop spec a partial-block access disables stays a
        candidate (``ConfigSpec.cute_reflatten_candidates``) for the device-IR
        analysis to re-register once the kernel proves pointwise."""
        return False

    def tunes_lane_vector_widths(self) -> bool:
        """Whether the search surface registers a vector-width slot
        (``ConfigSpec.cute_vector_widths``) for every static-size tile block,
        so a lane loop without one is a bug rather than a scalar lane."""
        return False

    def thread_axis_limit(self) -> int | None:
        """Number of thread axes a launch has, so the per-thread N-D strategy
        demotes the blocks beyond them to single-thread lane loops; ``None``
        never demotes."""
        return None

    def record_launch_thread_extent(
        self, fn: DeviceFunction, name: str, axis: int, extent: int
    ) -> None:
        """Note that the index variable ``name`` of a tile lives on thread
        ``axis`` of the launch, ``extent`` threads wide (the launch shape)."""
        return None

    def record_grid_thread_extent(
        self, fn: DeviceFunction, name: str, axis: int, extent: int
    ) -> None:
        """Note that ``name`` is a root tile's own coordinate on thread
        ``axis``, ``extent`` wide (the evidence a pointwise root's producer
        axes are read from)."""
        return None

    def referenced_thread_axes(self, statements: Sequence[ast.AST]) -> list[int]:
        """The launch thread axes ``statements`` read (an axis per reference,
        repeats included), so the code generator can grow the launch block to
        the extent of every axis a statement touches.  A backend whose
        per-thread code reads thread coordinates reports them; the default has
        none."""
        return []

    def record_vloop_wrapper(
        self,
        fn: DeviceFunction,
        wrapper: VecLaneWrapper,
        *,
        block_id: int,
        lane_var: str,
        index_var: str,
        mask_var: str | None,
        begin: object,
        extent: object,
        step: object,
    ) -> None:
        """Note the outer-lane x constexpr-V partition ``wrapper`` a strategy
        built for ``block_id``: ``index_var`` / ``mask_var`` are the per-element
        index and mask inside it, and ``begin`` / ``extent`` / ``step`` the
        tile's range (whether every V-wide chunk starts aligned and stays
        inside the extent follows from them)."""
        return None

    def apply_lane_cluster_split(
        self, strategy: PerThreadNDTileStrategy, state: CodegenState
    ) -> None:
        """Split the lane-looped axis of ``strategy`` across the CTAs of a
        thread-block cluster when the config asks for one and the loop
        qualifies; runs once per strategy, before its first device loop."""
        return None

    def check_lane_loop_nest(self, state: DeviceLoopState) -> None:
        """Make the lane loop nest a device loop built around its body the
        tile program (redistribute or reject an inexact nest, order the
        accesses threads could race on), once the body is complete."""
        return None

    def add_lane_nest_barriers(
        self,
        state: DeviceGridState,
        placement: list[ast.AST | LanePlacement],
        scopes: Sequence[LaneScope],
        rename_groups: Mapping[str, str],
    ) -> None:
        """Order across threads the accesses of a root body placed around its
        lane loops (``placement``, described by ``scopes``), in place."""
        return None

    def demote_grid_tile_vec_stores(
        self,
        state: DeviceGridState,
        body: list[ast.AST],
        placement: list[ast.AST | LanePlacement] | None,
    ) -> bool:
        """Restore to scalar form the collected tile-vector stores of a root
        body that a later statement of their lane loop observes
        (``placement`` is the body placed around the loops, or ``None`` for
        the full nest).  True when a store was demoted and the body must be
        placed again."""
        return False

    def demote_loop_tile_vec_stores(
        self, strategy: TileStrategy, block_ids: list[int]
    ) -> None:
        """Restore the tile-vector stores of a device loop's vectorized lanes
        (``block_ids``) that a later statement of the loop body observes,
        once the body is complete."""
        return None

    # -- codegen pipeline, backend state and registries ------------------------

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        """Run backend-specific passes after tiling is finalized, before codegen.

        Backends can override this to analyze or transform the graphs.
        """
        return None

    def finalize_device_body(
        self, body: list[ast.stmt], ctx: DeviceBodyContext
    ) -> tuple[list[ast.stmt], list[ast.arg]]:
        """Rewrite the rendered kernel body before variable renaming.

        Called by ``DeviceFunction.codegen_function_def`` once every statement
        of the kernel body (dependent-launch prologue, preambles, body) is
        rendered and before ``ast_rename`` applies the loop-carried aliases.
        Returns the body and the kernel parameters to append to the signature
        (``ctx.args`` is the signature so far).  Facts for
        ``finalize_kernel_def`` go on ``ctx``.  The default changes nothing.
        """
        return body, []

    def finalize_kernel_def(
        self,
        fn_def: ast.FunctionDef,
        prefix: list[ast.stmt],
        ctx: DeviceBodyContext,
    ) -> list[ast.stmt]:
        """Rewrite the renamed kernel ``FunctionDef`` in place and return the
        module-level statements that follow it.

        Called after ``ast_rename``; ``prefix`` holds the module-level
        statements emitted ahead of the definition.  The default changes
        nothing and returns ``[]``.
        """
        return []

    def validate_codegen_config(
        self, env: CompileEnvironment, device_ir: DeviceIR, config: Config
    ) -> None:
        """Validate ``config`` against ``device_ir`` before the top-level
        kernel is generated (not for the codegen graphs a caller supplies);
        raise ``InvalidConfig`` for a request the kernel cannot honour."""
        return None

    def after_device_codegen(
        self, codegen: GenerateAST, host: HostFunction, config: Config
    ) -> None:
        """Run once every host statement has been visited and the device body is
        complete, before ``codegen_function_def``: deferred checks and the
        device-side additions that depend on the whole body."""
        return None

    def prepare_device_loop(
        self,
        graph_info: ForLoopGraphInfo,
        state: CodegenState,
        device_loop: DeviceLoopState,
    ) -> None:
        """Run before the body of the device loop ``graph_info`` is rendered
        into ``device_loop`` (checks over the loop's carries, planning that
        the body's lowering consults).  No-op by default."""
        return None

    def finish_device_loop_body(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        outputs: list[object],
        initials: dict[int, str],
    ) -> None:
        """Run after the body of ``device_loop`` is rendered and before its
        carried outputs are recorded: ``outputs`` are the loop's outputs (the
        phi after the loop merges ``outputs[i]`` with ``initials[i]``) and may
        be rewritten in place.  No-op by default."""
        return None

    def split_persistent_post_loop(
        self, fn: DeviceFunction, body: list[ast.AST]
    ) -> tuple[list[ast.AST], list[ast.stmt]]:
        """Split the statements a persistent kernel runs once after its tile
        loop (one-shot teardown the body's lowering registered) out of the
        loop body ``body``; ``(body, [])`` by default."""
        return body, []

    def collective_sibling_blocks(
        self, env: CompileEnvironment, config: Config
    ) -> tuple[set[int], set[tuple[int, ...]]] | None:
        """The blocks a collective (hardware-cooperative) matmul owns and the
        root loops that hold one, when ``config`` selects such a lowering:
        ``TileStrategyDispatch`` then fits the independent SIMT loops of other
        roots to the collective's physical launch.  ``None`` by default."""
        return None

    def finalize_host_statements(
        self, codegen: GenerateAST, host: HostFunction, statements: list[ast.AST]
    ) -> tuple[list[ast.AST], list[ast.AST]]:
        """Rewrite the host wrapper's statements after dead-code elimination;
        returns them with the statements to emit after the kernel's module
        metadata (``[]`` by default).  A backend that annotates the kernel
        object from the host (CuTe's wrapper plans, cluster shape and
        shape-bake exemption) renders those statements here."""
        return statements, []

    def synthetic_thread_axis_sizes(self, codegen: GenerateAST) -> Mapping[int, int]:
        """The thread extent along each launch axis the body being generated
        addresses outside its loop strategies (CuTe: the free ``hl.arange``
        dims' synthetic axes); ``GenerateAST.active_thread_axis_sizes`` folds
        them in.  Empty by default."""
        return {}

    def consume_root_lane_loop_suppression(self, fn: DeviceFunction) -> bool:
        """Whether the root body about to be emitted drops its lane loops: a
        one-shot request a collective lowering of ``fn`` registered because it
        owns the lanes itself.  ``False`` by default."""
        return False

    def reorder_post_loop_statements(
        self, fn: DeviceFunction, body: list[ast.AST]
    ) -> list[ast.AST]:
        """Order the flat-grid kernel ``body`` so the statements the backend
        tagged as post-loop cleanup run last (the persistent path pulls them
        out of its work loop through ``split_persistent_post_loop``).
        ``body`` unchanged by default."""
        return body

    def roots_fissioned(self, env: CompileEnvironment) -> bool:
        """Whether the backend launches this kernel's root loops separately
        itself (CuTe's materialized fission), so the shared-program-id rules
        of a multi-root kernel do not apply.  ``False`` by default."""
        return False

    # Backend state.  Each core compile object carries one opaque
    # ``backend_state`` slot the backend fills through these hooks and reads
    # back through its own typed accessors; core never names the fields.  The
    # defaults are ``None``: a backend without per-compile state leaves the
    # slots empty, and core reaches a backend's state only through hooks.

    def new_env_state(self) -> object | None:
        """State stored in ``CompileEnvironment.backend_state`` for the compile."""
        return None

    def new_codegen_state(self) -> object | None:
        """State stored in ``GenerateAST.backend_state`` for one codegen pass."""
        return None

    def new_device_state(self, config: Config) -> object | None:
        """State stored in ``DeviceFunction.backend_state`` for one device
        function rendered under ``config``."""
        return None

    def root_emitters(self) -> tuple[RootEmitter, ...]:
        """Special-purpose emitters tried in order for each root loop before
        its ordinary lowering (``GenerateAST.visit_For``)."""
        return ()

    def kernel_generators(self) -> tuple[KernelGenerator, ...]:
        """Whole-kernel generators tried in order before the ordinary
        ``GenerateAST`` pass (``_generate_ast``)."""
        return ()

    def retry_policy(self) -> RetryPolicy:
        """The regeneration signals ``generate_ast`` retries on and how each
        pass repairs the config."""
        return NO_RETRY

    def decomp_table_overrides(
        self, table: dict[torch._ops.OpOverload, Callable[..., object]]
    ) -> None:
        """Adjust the Inductor decomposition table used while tracing.

        Called on the copied table before Helion's own overrides are applied;
        a backend that lowers an op natively pops its decomposition here.
        Default: no-op.
        """
        return

    def rewrite_device_ir(
        self,
        stage: DeviceIRStage,
        func: HostFunction | None,
        device_ir: DeviceIR,
        analysis: DeviceIRAnalysis | None = None,
    ) -> None:
        """Run the backend's device-IR passes for ``stage``.

        Called at most once per :class:`DeviceIRStage` during
        ``lower_to_device_ir``, in enum order (``func`` is the owning
        ``HostFunction``; ``analysis`` is the shared ``DeviceIRAnalysis`` from
        ``AFTER_ANALYSIS`` on).  Passes may rewrite
        the graphs or the host AST and register config-spec slots; they must
        not reorder themselves across stages.  Default: no-op.
        """
        return

    def register_search_surface(
        self, device_ir: DeviceIR, config_spec: ConfigSpec
    ) -> None:
        """Enable the backend's structural autotune knob families.

        Called once after ``register_rollable_reductions`` and before the
        multi-root pid restriction; the backend detects the kernel shapes its
        special-purpose emitters cover and widens ``config_spec`` for them.
        Default: no-op.
        """
        return

    # -- host wrapper arguments ------------------------------------------------

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        """Names reserved by this backend's kernel launch mechanism.

        These names cannot be used as kernel variables because they
        collide with parameters of the backend's kernel launch API
        (e.g., Triton's ``run()`` method uses ``grid``, ``num_warps``,
        ``num_stages``, etc.).
        """
        return frozenset()

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        """Transform a host argument expression before passing to the launcher.

        Backends can override this to wrap certain argument types.
        Called during codegen for each argument in sorted order.
        """
        return host_str

    def tensor_descriptor_host_base(
        self, fake_value: torch.Tensor, host_str: str
    ) -> str:
        """Return the host tensor expression used to construct a descriptor."""
        return host_str

    def tensors_proven_disjoint(
        self, env: CompileEnvironment, left: torch.Tensor, right: torch.Tensor
    ) -> bool:
        """Whether two input tensors behind kernel arguments are proven not to
        alias by a runtime fact that is part of the bound kernel's cache key
        (``DeviceFunction.proven_disjoint_tensor_pairs``).  ``False`` by
        default: no backend-owned runtime storage fact."""
        return False

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        """Generate preamble statements for scalar arguments in the device function.

        Backends can override to dereference scalar refs, etc.
        """
        return []

    def rng_seed_buffer_expr(self, count: int) -> str:
        """Return the Python expression string that creates the RNG seed buffer.

        Backends can override to customize seed generation (e.g. for devices
        that don't support int64 randint).
        """
        return f"inductor_prims.seeds({count}, torch.accelerator.current_accelerator())"

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
        """The argument list of the generated launch call: ``args`` (the kernel's
        positional arguments) followed by ``launcher_keyword_args``.
        ``tensor_host_args`` / ``sorted_args`` let a backend add host-side tensor
        metadata; kernels with RNG ops are rejected by default."""
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    # -- runtime hooks (``helion.runtime.kernel``) ---------------------------
    # Each hook sits at one fixed position of the bind / render / compile
    # sequence; the defaults do nothing.

    def on_bind(
        self,
        bound: BoundKernel[Any],
        args: tuple[object, ...],
        *,
        rewrite_ast: bool,
    ) -> None:
        """Plan backend-owned rewrites of a kernel being bound.

        Called from ``BoundKernel.__init__`` after its ``CompileEnvironment``
        exists and before the host function is traced; ``rewrite_ast`` is the
        bind's flag of the same name.
        """
        return None

    def host_structural_facts(
        self, host_function: HostFunction, config_spec: ConfigSpec
    ) -> None:
        """Record structural facts about the traced host function on the spec.

        Called once per bind, inside the compile environment and before the
        compiler seeds are generated, so seeds and heuristics read final facts.
        """
        return None

    def before_codegen(self, bound: BoundKernel[Any], config: Config) -> None:
        """Runs in ``BoundKernel.to_code`` right before ``generate_ast``."""
        return None

    def after_codegen(self, bound: BoundKernel[Any], config: Config) -> None:
        """Runs in ``BoundKernel.to_code`` after a successful render, once the
        guarded input facts are published; backends register the
        specializations their generated code depends on here."""
        return None

    def after_compile(self, bound: BoundKernel[Any], config: Config) -> None:
        """Runs in ``BoundKernel.compile_config`` whenever a compiled callable
        is handed out (fresh or from the in-memory cache)."""
        return None

    def decorator_settings(self, settings: Settings) -> list[str]:
        """Extra ``name=value`` parts of the ``@helion.kernel`` repro decorator.

        Resolved from ``settings.backend`` (not from a compile environment) by
        ``BoundKernel.format_kernel_decorator``; the parts follow ``config``,
        ``static_shapes`` and ``index_dtype`` in the printed decorator.
        """
        return []

    def structural_policy_type(self) -> type[StructuralPolicy] | None:
        """The backend's structural policy class, or None when its config
        schema does not depend on Settings (see
        :mod:`helion.runtime.structural_policy`)."""
        return None

    # -- strategy factories and the autotune entry point -----------------------

    def create_program_ids(
        self, fn: DeviceFunction, pid_type: str, block_ids: list[int]
    ) -> ProgramIDs | None:
        """A backend-owned program-id strategy for the grid ``block_ids``
        under ``pid_type``, or ``None`` to use the generic strategies.

        Called by ``BlockSizeTileStrategy.select_pid_strategy`` after the
        compact-worklist and ``xyz`` selections; the result is wrapped by
        ``L2GroupingProgramIDs`` like any generic strategy.
        """
        return None

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        """The ``TileStrategy`` driving the device loop over ``block_ids`` under
        ``config``: flattened when the first block size is flattened, else N-D
        with the loop order and L2 grouping of ``config``.  Backends with their
        own strategies override this."""
        from .compile_environment import CompileEnvironment
        from .tile_strategy import FlattenedTileStrategy
        from .tile_strategy import NDTileStrategy

        env = CompileEnvironment.current()
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        # A registered tile can also appear in a multidimensional loop.
        # Its permutation applies only there; a one-dimensional loop has
        # no order to configure.
        loop_order = (
            env.config_spec.loop_orders.config_get(config.loop_orders, block_ids[0])
            if len(block_ids) > 1
            else None
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )

        if block_size_infos[0].is_flattened(config):
            block_size = functools.reduce(  # pyrefly: ignore[incompatible-overload-residual]
                operator.mul,
                [bs.from_config_assert(config) for bs in block_size_infos],
            )
            return FlattenedTileStrategy(
                fn,
                block_ids,
                block_size=block_size,
                loop_order=loop_order,
            )

        return NDTileStrategy(
            fn,
            block_ids,
            block_size=[bs.from_config_assert(config) for bs in block_size_infos],
            loop_order=loop_order,
            l2_grouping=l2_grouping,
        )

    def create_reduction_strategy(
        self,
        fn: DeviceFunction,
        block_id: int,
        reduction_loop: int | None,
    ) -> TileStrategy:
        """Create a reduction strategy for the given block dimension.

        Analogous to create_loop_strategy() but for reduction dimensions.
        Backends can override to return backend-specific strategy subclasses.
        """
        from .reduction_strategy import LoopedReductionStrategy
        from .reduction_strategy import PersistentReductionStrategy

        if reduction_loop is None:
            return PersistentReductionStrategy(fn, block_id)
        return LoopedReductionStrategy(fn, block_id, reduction_loop)

    def create_block_reduction_strategy(
        self,
        state: CodegenState,
        block_id: int,
    ) -> BlockReductionStrategy:
        """Create the strategy for a reduction over a tile block rather than a
        reduction dimension.

        Backends can override to return a subclass.
        """
        from .reduction_strategy import BlockReductionStrategy

        return BlockReductionStrategy(state, block_id)

    def reduction_emitter(
        self, strategy: ReductionStrategy
    ) -> ReductionEmitter[Any] | None:
        """The backend's emitter for ``strategy`` (``ReductionStrategy.emitter``).

        A thread-level backend returns the object that plans the reduction's
        thread layout and emits its cross-thread combines; ``None`` leaves
        every hook of the generic strategy classes on their default path.
        """
        return None

    def lower_owned_lane_loop(self, codegen: GenerateAST) -> None:
        """Lower the owned lane loops of a finished root body in place.

        Runs when ``GenerateAST`` leaves a root loop, after the persistent
        wrapper and before dead-code elimination, over
        ``codegen.device_function.body``.  A backend whose reduction
        strategies leave ownership markers inside per-thread lane loops
        rewrites them here (the two-pass accumulate / finalize / consume
        split, the lane / serial-loop interchange, the chunk-recurrence
        hoist); the default leaves the body as rendered.
        """
        return None

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        """Run autotuning to find the best configuration.

        This default implementation handles:
        - Using a single provided config directly
        - Searching over finite predetermined configs
        - Running a full search algorithm

        Subclasses can override to customize behavior (e.g., disabling
        precompile for backends that don't support it).
        """
        force = force or bound_kernel.settings.force_autotune

        # Backends without a precompile pool keep only the fork mode, and only
        # when they render candidates in forked workers instead.
        bound_kernel.settings.autotune_precompile = autotune_precompile_mode(
            self, bound_kernel.settings.autotune_precompile
        )

        if bound_kernel.settings.autotune_effort == "none" and (
            force or not bound_kernel.kernel.configs
        ):
            return bound_kernel.config_spec.default_config()
        if not force and bound_kernel.kernel.configs:
            if len(bound_kernel.kernel.configs) == 1:
                (config,) = bound_kernel.kernel.configs
                return config
            # We have finite predetermined configs, no need to precompile
            bound_kernel.settings.autotune_precompile = None

            from ..autotuner import FiniteSearch

            autotuner = FiniteSearch(bound_kernel, args, bound_kernel.configs)
        else:
            bound_kernel.settings.check_autotuning_disabled()
            autotuner = bound_kernel.settings.autotuner_fn(bound_kernel, args, **kwargs)
        if bound_kernel.settings.autotune_handoff:
            from ..autotuner.handoff_pipeline import autotune_with_handoff

            return autotune_with_handoff(autotuner, skip_cache=force)
        return autotuner.autotune(skip_cache=force)

    @staticmethod
    def map_dot_precision(precision: DotPrecision) -> str:
        """Map Helion dot precision to backend-specific precision string.

        Default implementation maps to Triton-compatible precision values.
        """
        triton_precision_by_dot_precision = {
            "default": "tf32",
            "high": "tf32x3",
            "highest": "ieee",
            "tf32": "tf32",
            "tf32x3": "tf32x3",
            "ieee": "ieee",
        }
        return triton_precision_by_dot_precision.get(precision, "")


def _largest_divisor_at_most(size: int, limit: int) -> int:
    for divisor in range(limit, 0, -1):
        if size % divisor == 0:
            return divisor
    return 1


def _loop_contains_matmul(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    """Whether the device loop over ``block_ids`` (nested for-loops included)
    calls an aten matmul or ``hl.dot``."""
    from ..language._decorators import is_api_func
    from .device_ir import iter_loop_graph_nodes

    matmul_targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    return any(
        node.op == "call_function"
        and (
            node.target in matmul_targets
            or (is_api_func(node.target) and node.target.__name__ == "dot")
        )
        for node in iter_loop_graph_nodes(fn, block_ids)
    )


def _active_loop_block_ids(fn: DeviceFunction) -> set[int]:
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    active: set[int] = {
        block_id for block_ids in device_ir.grid_block_ids for block_id in block_ids
    }
    for graph_info in fn.codegen.codegen_graphs:
        if isinstance(graph_info, ForLoopGraphInfo):
            active.update(graph_info.block_ids)
    return active


def autotune_precompile_mode(
    backend: Backend, requested: PrecompileMode
) -> PrecompileMode:
    """The ``autotune_precompile`` mode ``backend`` honours.

    A backend with a precompile pool takes the setting as is. One without
    keeps ``"fork"`` only to render candidates in forked workers
    (``Backend.supports_parallel_render``); ``"spawn"`` has no equivalent
    there and turns into ``None`` like any other mode. A function rather
    than a method so any object with ``supports_precompile`` (test doubles
    included) can stand in for a backend that has a pool.
    """
    if backend.supports_precompile():
        return requested
    if requested == "fork" and backend.supports_parallel_render():
        return "fork"
    return None
