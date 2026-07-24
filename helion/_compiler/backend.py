from __future__ import annotations

import abc
import dataclasses
import functools
import logging
import math
import operator
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Sequence

import sympy
import torch

from .. import exc
from .ast_extension import expr_from_string
from .cute.attention_plan import ALIBI_BIAS_KIND
from .cute.attention_plan import CAUSAL_MASK_KIND
from .cute.attention_plan import DOCUMENT_MASK_KIND
from .cute.attention_plan import PREFIX_LM_MASK_KIND
from .cute.attention_plan import RELATIVE_BIAS_KIND
from .cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from .cute.attention_plan import SOFTCAP_KIND
from .cute.attention_plan import TENSOR_BIAS_KIND
from .cute.attention_plan import AttentionScoreModifier
from .cute.attention_plan import AttentionScorePlan

if TYPE_CHECKING:
    import ast
    import contextlib

    from torch._inductor.ops_handler import OpsHandler

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..autotuner.config_priors import ValuePrior
    from ..autotuner.config_spec import ConfigSpec
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.settings import DotPrecision
    from .cute.cute_mma import _CuteMmaNode
    from .device_function import Argument
    from .device_function import DeviceFunction
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .host_function import HostFunction
    from .tile_dispatch import TileStrategyDispatch
    from .tile_strategy import TileStrategy

    InductorOpOverrides = OpsHandler[Any]

log: logging.Logger = logging.getLogger(__name__)


class FlashSearchSurface(NamedTuple):
    head_dim: int
    num_kv: int
    block_size_targets: dict[int, int]
    is_causal: bool
    has_kv_tile_pruning: bool
    requires_ws_overlap: bool
    small_biased_candidate: bool


class AttentionSoftmaxPattern(NamedTuple):
    score_plan: AttentionScorePlan

    @property
    def head_dim(self) -> int:
        return self.score_plan.head_dim

    @property
    def is_causal(self) -> bool:
        return self.score_plan.is_causal


class Backend(abc.ABC):
    """Abstract base class for Helion code generation backends.

    Each backend is responsible for defining:
    - How types are represented in generated code
    - What imports are needed in generated code
    - What decorators and annotations are used on generated functions
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend name used for codegen dispatch (e.g., 'triton')."""
        ...

    @property
    def experimental(self) -> bool:
        """Whether this backend is experimental and should emit a warning."""
        return True

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
        raise exc.BackendUnsupported(self.name, "program IDs")

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
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

    def max_reduction_threads(self) -> int | None:
        """Maximum threads for a single warp-level reduction, or None if unlimited."""
        return None

    def max_reduction_loop(self) -> int | None:
        """Maximum user-visible loop chunk for a rolled reduction."""
        return self.max_reduction_threads()

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        """Adjust reduction thread count to fit within hardware thread limits.

        Tile-level backends return the count unchanged. Thread-level backends
        (e.g., CuTe) override this to cap against the per-block thread budget
        shared across all tiled dimensions.
        """
        return requested

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

    def barrier_semaphore_dtype(self) -> torch.dtype:
        """Dtype used for persistent multi-phase barrier semaphore tensors."""
        return torch.uint32

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        """Statement emitted between persistent phases, if supported."""
        raise exc.BackendUnsupported(self.name, "hl.barrier()")

    def reduction_axis_first(self) -> bool:
        """Whether reduction strategies should occupy the first (lowest) thread axes."""
        return False

    def force_tile_mask(self) -> bool:
        """Whether tile strategies must emit explicit masks for all tiles."""
        return False

    def supports_config_key(self, key: str) -> bool:
        from ..autotuner.config_spec import BACKEND_SPECIFIC_KEYS

        return key not in BACKEND_SPECIFIC_KEYS

    def supports_block_ptr_indexing(self) -> bool:
        return True

    def process_fake_tensor_load(
        self,
        tensor: torch.Tensor,
        index: list[object],
    ) -> None:
        """Called during `type_propagation` when processing a `load` memory op on fake tensors"""
        return

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

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        return {}

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]] | None:
        """Return the benchmarking function for this backend.

        The default returns ``None`` which causes the autotuner to use the
        module-level ``do_bench`` (patchable by tests).  Backends that need
        a different timing mechanism (e.g., Pallas/TPU) should override
        this to return their own function.
        """
        return None

    def get_interleaved_bench(
        self,
    ) -> Callable[..., list[float]] | None:
        """Return the interleaved benchmarking function for this backend.

        The default returns ``None`` which causes the autotuner to use the
        module-level ``interleaved_bench``.  Backends without Triton event
        timing should override.
        """
        return None

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
        raise exc.BackendUnsupported(self.name, "full tensor creation")

    def reshape_expr(self, expr: str, shape: str) -> str:
        raise exc.BackendUnsupported(self.name, "reshape")

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
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
    ) -> str:
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

    def reduction_index_init_expr(
        self, shape_dims: list[str], index_dtype: torch.dtype
    ) -> str:
        """Initial accumulator value for index-carrying reductions."""
        return self.full_expr(
            shape_dims, repr(torch.iinfo(index_dtype).max), index_dtype
        )

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
    ) -> str:
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        raise exc.BackendUnsupported(self.name, "argmin/argmax reductions")

    def inductor_op_overrides(self) -> InductorOpOverrides:
        raise exc.BackendUnsupported(self.name, "Inductor OpOverrides")

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            self.cast_expr("{x}", self.dtype_str(target_dtype)),
            x=x,
        )

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

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        return []

    def customize_ast(self, hf: HostFunction) -> None:
        """Run backend-specific AST customizations.

        Called after static loop unrolling but before type propagation
        and tracing.  Backends can override this to rewrite the user's
        AST for algorithmic transformations that change loop structure.
        """
        return None

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
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    def _cute_matmul_contraction_reduction_block_ids(self) -> set[int]:
        """Reduction block ids that are also a matmul-contraction (K) axis.

        These are the blocks that must keep real threads for the whole K extent
        instead of being split into ``threads x synthetic-lane`` (see OPTION B in
        ``CuteBackend.create_loop_strategy`` and
        ``cute_matmul_contraction_block_ids``).
        """
        from .compile_environment import CompileEnvironment
        from .cute.matmul_utils import cute_matmul_contraction_block_ids

        env = CompileEnvironment.current()
        canonical_block_id = getattr(
            env, "canonical_block_id", lambda block_id: block_id
        )
        contraction = cute_matmul_contraction_block_ids()
        if not contraction:
            return set()
        return {
            info.block_id
            for info in env.block_sizes
            if info.reduction and canonical_block_id(info.block_id) in contraction
        }

    def _cute_matmul_contraction_thread_reserve(
        self, fn: DeviceFunction, tile_block_ids: list[int]
    ) -> int:
        """Threads to reserve for matmul-contraction reduction axes.

        Returns the product of the per-axis full thread extents (power-of-two,
        capped at ``max_reduction_threads``) of every reduction block that is a
        matmul-contraction axis and is *not* one of ``tile_block_ids`` (i.e. it
        is handled by a separate reduction strategy, not this tile strategy).
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        from .._compat import shape_env_size_hint
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        max_reduction_threads = self.max_reduction_threads()
        if max_reduction_threads is None:
            return 1
        tile_ids = set(tile_block_ids)
        reserve = 1
        for block_id in self._cute_matmul_contraction_reduction_block_ids():
            if block_id in tile_ids:
                continue
            numel = env.block_sizes[block_id].numel
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            if size_hint <= 1:
                continue
            reserve *= next_power_of_2(min(size_hint, max_reduction_threads))
        return reserve

    def _cute_free_auto_thread_axis_count(
        self, fn: DeviceFunction, config: Config
    ) -> int:
        """Count the kernel's free (non-reduction) tile axes that auto-thread.

        These are the axes that compete for the thread budget left over after a
        matmul-contraction reduction has reserved its slice (see OPTION B in
        ``create_loop_strategy``).  The reserve's ``thread_limit`` shrink is
        applied per ``create_loop_strategy`` call, but a kernel may build the M
        and N tile axes in *separate* calls (e.g. M is the grid, N is a device
        loop).  Each call only sees its own axes, so dividing the per-call
        ``thread_limit`` by the reserve once is not enough: the product of every
        free axis' threads must stay within ``1024 // reserve``.  Counting all
        the free auto-threaded axes lets each call take only its fair share.
        """
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        count = 0
        for block_id in _active_loop_block_ids(fn):
            info = env.block_sizes[block_id]
            if info.reduction:
                continue
            block_size = info.from_config(config)
            if not isinstance(block_size, int) or block_size <= 1:
                continue
            threads = int(
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
            )
            # Only auto-threaded (``num_threads == 0``) axes participate in the
            # budget split; explicitly-threaded axes keep their configured count.
            if threads == 0:
                count += 1
        return max(count, 1)

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from .compile_environment import CompileEnvironment
        from .tile_strategy import FlattenedTileStrategy
        from .tile_strategy import NDTileStrategy

        env = CompileEnvironment.current()
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )

        if block_size_infos[0].is_flattened(config):
            block_size = functools.reduce(  # pyrefly: ignore[incompatible-overload-residual]
                operator.mul, [bs.from_config_assert(config) for bs in block_size_infos]
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

        # Disable precompile for backends that don't support it
        if not self.supports_precompile():
            bound_kernel.settings.autotune_precompile = None

        if bound_kernel.settings.autotune_effort == "none" and (
            force or not bound_kernel.kernel.configs
        ):
            config = bound_kernel.config_spec.default_config()
        elif not force and bound_kernel.kernel.configs:
            if len(bound_kernel.kernel.configs) == 1:
                (config,) = bound_kernel.kernel.configs
            else:
                # We have finite predetermined configs, no need to precompile
                bound_kernel.settings.autotune_precompile = None

                from ..autotuner import FiniteSearch

                config = FiniteSearch(
                    bound_kernel, args, bound_kernel.configs
                ).autotune()
        else:
            bound_kernel.settings.check_autotuning_disabled()
            config = bound_kernel.settings.autotuner_fn(
                bound_kernel, args, **kwargs
            ).autotune(skip_cache=force)
        return config

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


# TPU does not natively support 64-bit element types.
_PALLAS_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


def _largest_divisor_at_most(size: int, limit: int) -> int:
    for divisor in range(limit, 0, -1):
        if size % divisor == 0:
            return divisor
    return 1


def _specialized_mma_root_mn_block_ids(
    candidate: _CuteMmaNode,
    config: Config,
) -> tuple[int, int] | None:
    """Return exact root-grid matrix axes for an analyzed MMA candidate."""
    from .compile_environment import CompileEnvironment
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return None
    operands = candidate.operands
    if tuple(device_ir.grid_block_ids[0]) != operands.output_block_ids:
        return None
    if (leading_id := operands.leading_passthrough_block_id) is not None:
        block_size = (
            CompileEnvironment.current().block_sizes[leading_id].from_config(config)
        )
        if not isinstance(block_size, int) or block_size != 1:
            return None
    return operands.m_block_id, operands.n_block_id


def _attention_flash_gate_enabled() -> bool:
    """Gate for the fused tcgen05 flash-attention path.

    The fused QK->softmax->PV tcgen05 codegen (mirrors
    ``.notes/spikes/fa_tcgen05_spike.py``) is ON by default: a detected fp16/bf16,
    128x128, dense or canonical-causal, seq%128==0 attention kernel runs on the
    tensor cores. Set ``HELION_CUTE_FLASH=0`` to force the scalar-fallback path.
    The detector (``_attention_loop_shape`` plus graph-metadata output
    validation) is strict, so any config outside the envelope (fp32, unsupported
    score masks, head_dim not in {64,128}, non-128 tiles, seq%128!=0) falls back
    to the scalar path with no behavior change.
    """
    return os.environ.get("HELION_CUTE_FLASH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _attention_flash_supported() -> bool:
    from .cute.mma_support import get_cute_mma_support

    return bool(get_cute_mma_support().tcgen05_f16bf16)


def _attention_loop_carried_arg(
    graph: torch.fx.Graph,
    index: int,
) -> torch.fx.Node | None:
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if index >= len(placeholders):
        return None
    return placeholders[index]


def _attention_new_var_source(node: torch.fx.Node) -> torch.fx.Node | None:
    from ..language._tracing_ops import _new_var

    if node.op != "call_function" or node.target is not _new_var:
        return None
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    return node.args[0]


def _attention_is_loop_carried_value(
    node: torch.fx.Node,
    placeholder: torch.fx.Node,
) -> bool:
    return node is placeholder or _attention_new_var_source(node) is placeholder


def _attention_is_full_slice(value: object) -> bool:
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _attention_is_block_symnode(node: torch.fx.Node, block_id: int) -> bool:
    from ..language._tracing_ops import _get_symnode

    return (
        node.op == "call_function"
        and node.target is _get_symnode
        and len(node.args) >= 1
        and node.args[0] == f"block_size_{block_id}"
    )


def _attention_is_inner_batch_index(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_size.int
        and len(node.args) >= 2
        and isinstance(node.args[1], int)
        and node.args[1] == 0
    )


def _attention_arg_is_scaled_q(
    node: torch.fx.Node,
    expected_scale: float,
    q_placeholder: torch.fx.Node,
) -> bool:
    scale = _attention_arg_scaled_q_factor(node, q_placeholder)
    return scale is not None and math.isclose(
        scale, expected_scale, rel_tol=1e-5, abs_tol=1e-7
    )


def _attention_arg_scaled_q_factor(
    node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
) -> float | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.mul.Tensor:
        return None
    if len(node.args) < 2:
        return None
    tensor_arg = None
    scale_arg = None
    for arg in node.args[:2]:
        if isinstance(arg, torch.fx.Node):
            tensor_arg = arg
        elif isinstance(arg, (int, float)):
            scale_arg = float(arg)
    if (
        tensor_arg is None
        or scale_arg is None
        or not _attention_is_loop_carried_value(tensor_arg, q_placeholder)
    ):
        return None
    return scale_arg


def _attention_pv_p_arg_base(node: torch.fx.Node) -> torch.fx.Node:
    from ..language._tracing_ops import _mask_to

    while (
        node.op == "call_function"
        and node.target
        in (
            _mask_to,
            torch.ops.prims.convert_element_type.default,
        )
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    return node


def _attention_is_negative_infinity(node: object) -> bool:
    if isinstance(node, (float, int)):
        return float(node) == float("-inf")
    if not isinstance(node, torch.fx.Node):
        return False
    if (
        node.op == "call_function"
        and node.target is torch.ops.aten.scalar_tensor.default
    ):
        value = node.args[0] if node.args else None
        return isinstance(value, (float, int)) and float(value) == float("-inf")
    return False


def _attention_tile_index_source(node: object) -> torch.fx.Node | None:
    from ..language import memory_ops
    from ..language import tile_ops
    from ..language import view_ops

    while isinstance(node, torch.fx.Node):
        if node.op != "call_function":
            return None
        if node.target in (memory_ops.load, view_ops.subscript):
            if not node.args or not isinstance(node.args[0], torch.fx.Node):
                return None
            node = node.args[0]
            continue
        if node.target is tile_ops.tile_index:
            source = node.args[0] if node.args else None
            return source if isinstance(source, torch.fx.Node) else None
        return None
    return None


def _attention_is_q_tile_index(
    node: object,
    q_placeholder: torch.fx.Node,
) -> bool:
    source = _attention_tile_index_source(node)
    if source is None or len(source.args) < 2:
        return False
    dim = source.args[1]
    return (
        source.op == "call_function"
        and source.target is torch.ops.aten.sym_size.int
        and source.args[0] is q_placeholder
        and isinstance(dim, int)
        and dim == 1
    )


def _attention_is_kv_tile_index(node: object, kv_block_id: int | None) -> bool:
    source = _attention_tile_index_source(node)
    if source is None:
        return False
    if kv_block_id is None:
        return True
    return _attention_is_block_symnode(source, int(kv_block_id))


def _attention_causal_score_node(
    qk_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> torch.fx.Node | None:
    """Return the canonical causal ``where(tile_m >= tile_n, qk, -inf)`` node."""
    users = [
        user
        for user in qk_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.where.self
    ]
    if len(users) != 1 or set(qk_node.users) != {users[0]}:
        return None
    where_node = users[0]
    if len(where_node.args) < 3 or where_node.args[1] is not qk_node:
        return None
    if not _attention_is_negative_infinity(where_node.args[2]):
        return None
    pred = where_node.args[0]
    if (
        not isinstance(pred, torch.fx.Node)
        or pred.op != "call_function"
        or pred.target is not torch.ops.aten.ge.Tensor
        or len(pred.args) < 2
    ):
        return None
    lhs, rhs = pred.args[:2]
    if not _attention_is_q_tile_index(lhs, q_placeholder):
        return None
    if not _attention_is_kv_tile_index(rhs, kv_block_id):
        return None
    return where_node


def _attention_load_host_tensor_name(node: torch.fx.Node) -> str | None:
    from ..language import memory_ops
    from ..language._tracing_ops import _host_tensor

    if node.op != "call_function" or node.target is not memory_ops.load:
        return None
    tensor = node.args[0] if node.args else None
    if (
        isinstance(tensor, torch.fx.Node)
        and tensor.op == "call_function"
        and tensor.target is _host_tensor
        and tensor.args
        and isinstance(tensor.args[0], str)
    ):
        return tensor.args[0]
    return None


def _attention_is_q_block_index(
    node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_size.int
        and len(node.args) >= 2
        and node.args[0] is q_placeholder
        and isinstance(node.args[1], int)
        and node.args[1] == 1
    )


def _attention_tensor_bias_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    """Return ``score + bias[tile_b, tile_m, tile_n]`` as a score modifier."""
    from ..language import memory_ops

    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.add.Tensor
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    add_node = users[0]
    if add_node.kwargs.get("alpha", 1) != 1 or len(add_node.args) < 2:
        return None
    if add_node.args[0] is score_node and isinstance(add_node.args[1], torch.fx.Node):
        bias_node = add_node.args[1]
    elif add_node.args[1] is score_node and isinstance(add_node.args[0], torch.fx.Node):
        bias_node = add_node.args[0]
    else:
        return None
    if bias_node.op != "call_function" or bias_node.target is not memory_ops.load:
        return None
    tensor_name = _attention_load_host_tensor_name(bias_node)
    if tensor_name is None:
        return None
    if (
        len(bias_node.args) < 4
        or bias_node.args[2] is not None
        or bias_node.args[3] is not None
    ):
        return None
    indices = bias_node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    batch_idx, q_idx, kv_idx = indices
    if not isinstance(batch_idx, torch.fx.Node) or not isinstance(q_idx, torch.fx.Node):
        return None
    if not _attention_is_inner_batch_index(batch_idx):
        return None
    if not _attention_is_q_block_index(q_idx, q_placeholder):
        return None
    if not isinstance(kv_idx, torch.fx.Node):
        return None
    if kv_block_id is not None and not _attention_is_block_symnode(
        kv_idx, int(kv_block_id)
    ):
        return None
    return add_node, AttentionScoreModifier(TENSOR_BIAS_KIND, tensor_name=tensor_name)


def _attention_add_score_user(
    score_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.add.Tensor
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    add_node = users[0]
    if add_node.kwargs.get("alpha", 1) != 1 or len(add_node.args) < 2:
        return None
    if add_node.args[0] is score_node and isinstance(add_node.args[1], torch.fx.Node):
        return add_node, add_node.args[1]
    if add_node.args[1] is score_node and isinstance(add_node.args[0], torch.fx.Node):
        return add_node, add_node.args[0]
    return None


def _attention_scalar_value(node: object) -> float | None:
    if isinstance(node, (int, float)):
        return float(node)
    if not isinstance(node, torch.fx.Node):
        return None
    if (
        node.op == "call_function"
        and node.target is torch.ops.aten.scalar_tensor.default
        and node.args
        and isinstance(node.args[0], (int, float))
    ):
        return float(node.args[0])
    return None


class _AttentionLinearIndex(NamedTuple):
    q_coeff: float
    k_coeff: float
    constant: float


def _attention_index_linear_coeffs(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> _AttentionLinearIndex | None:
    """Return ``query_coeff * q + key_coeff * k + constant`` for ``node``."""
    scalar = _attention_scalar_value(node)
    if scalar is not None:
        return _AttentionLinearIndex(0.0, 0.0, scalar)
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if _attention_is_q_tile_index(node, q_placeholder):
        return _AttentionLinearIndex(1.0, 0.0, 0.0)
    if _attention_is_kv_tile_index(node, kv_block_id):
        return _AttentionLinearIndex(0.0, 1.0, 0.0)
    if node.target in (torch.ops.aten.sub.Tensor, torch.ops.aten.add.Tensor):
        if len(node.args) < 2:
            return None
        lhs = _attention_index_linear_coeffs(
            node.args[0],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        rhs = _attention_index_linear_coeffs(
            node.args[1],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if lhs is None or rhs is None:
            return None
        if node.target is torch.ops.aten.sub.Tensor:
            return _AttentionLinearIndex(
                lhs.q_coeff - rhs.q_coeff,
                lhs.k_coeff - rhs.k_coeff,
                lhs.constant - rhs.constant,
            )
        return _AttentionLinearIndex(
            lhs.q_coeff + rhs.q_coeff,
            lhs.k_coeff + rhs.k_coeff,
            lhs.constant + rhs.constant,
        )
    if node.target is torch.ops.aten.mul.Tensor and len(node.args) >= 2:
        lhs_scalar = _attention_scalar_value(node.args[0])
        rhs_scalar = _attention_scalar_value(node.args[1])
        if lhs_scalar is not None:
            rhs = _attention_index_linear_coeffs(
                node.args[1],
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            return (
                None
                if rhs is None
                else _AttentionLinearIndex(
                    lhs_scalar * rhs.q_coeff,
                    lhs_scalar * rhs.k_coeff,
                    lhs_scalar * rhs.constant,
                )
            )
        if rhs_scalar is not None:
            lhs = _attention_index_linear_coeffs(
                node.args[0],
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            return (
                None
                if lhs is None
                else _AttentionLinearIndex(
                    rhs_scalar * lhs.q_coeff,
                    rhs_scalar * lhs.k_coeff,
                    rhs_scalar * lhs.constant,
                )
            )
    return None


def _attention_index_delta_factor(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> float | None:
    """Return the coefficient of ``(query_index - key_index)`` in ``node``."""
    coeffs = _attention_index_linear_coeffs(
        node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if coeffs is None:
        return None
    if not math.isclose(coeffs.q_coeff + coeffs.k_coeff, 0.0, abs_tol=1e-7):
        return None
    if not math.isclose(coeffs.constant, 0.0, abs_tol=1e-7):
        return None
    return coeffs.q_coeff


def _attention_product_terms(node: object) -> list[object]:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is torch.ops.aten.mul.Tensor
        and len(node.args) >= 2
    ):
        return _attention_product_terms(node.args[0]) + _attention_product_terms(
            node.args[1]
        )
    return [node]


class _AttentionBatchIndex(NamedTuple):
    mode: str
    divisor: int | None = None


def _attention_alibi_slope_tensor_name(
    node: object,
) -> tuple[str, _AttentionBatchIndex] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_name = _attention_load_host_tensor_name(node)
    if tensor_name is None:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 1:
        return None
    index = indices[0]
    index_mode = _attention_collapsed_batch_index_mode(
        index,
        allow_mod=True,
        allow_floordiv=False,
    )
    if index_mode is None:
        return None
    return tensor_name, index_mode


def _attention_positive_int_scalar(node: object) -> int | None:
    value = _attention_scalar_value(node)
    if value is None:
        return None
    int_value = int(value)
    if int_value <= 0 or not math.isclose(value, float(int_value), abs_tol=1e-7):
        return None
    return int_value


def _attention_collapsed_batch_index_mode(
    node: object,
    *,
    allow_mod: bool,
    allow_floordiv: bool,
) -> _AttentionBatchIndex | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if _attention_is_inner_batch_index(node):
        return _AttentionBatchIndex("identity")
    source = _attention_tile_index_source(node)
    if source is not None and _attention_is_inner_batch_index(source):
        return _AttentionBatchIndex("identity")
    if node.op != "call_function" or len(node.args) < 2:
        return None
    if node.target in (
        operator.mod,
        torch.ops.aten.remainder.Scalar,
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_mod or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("mod", divisor)
    if node.target in (
        operator.floordiv,
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.floor_divide.Scalar,
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_floordiv or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("floordiv", divisor)
    if (
        node.target is torch.ops.aten.div.Tensor_mode
        and node.kwargs.get("rounding_mode") == "floor"
    ):
        lhs, rhs = node.args[:2]
        divisor = _attention_positive_int_scalar(rhs)
        if not allow_floordiv or divisor is None:
            return None
        lhs_mode = _attention_collapsed_batch_index_mode(
            lhs,
            allow_mod=False,
            allow_floordiv=False,
        )
        if lhs_mode is None or lhs_mode.mode != "identity":
            return None
        return _AttentionBatchIndex("floordiv", divisor)
    return None


def _attention_alibi_bias_modifier(
    bias_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> AttentionScoreModifier | None:
    terms = _attention_product_terms(bias_node)
    scalar = 1.0
    slope_name: str | None = None
    slope_index: _AttentionBatchIndex | None = None
    delta_factor: float | None = None
    for term in terms:
        term_scalar = _attention_scalar_value(term)
        if term_scalar is not None:
            scalar *= term_scalar
            continue
        term_slope = _attention_alibi_slope_tensor_name(term)
        if term_slope is not None:
            if slope_name is not None:
                return None
            slope_name, slope_index = term_slope
            continue
        term_delta = _attention_index_delta_factor(
            term,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if term_delta is None or math.isclose(term_delta, 0.0, abs_tol=1e-12):
            return None
        if delta_factor is not None:
            return None
        delta_factor = term_delta
    if slope_name is None or delta_factor is None:
        return None
    # Runtime lowers ALiBi as ``(key - query) * slope * scale``.
    return AttentionScoreModifier(
        ALIBI_BIAS_KIND,
        tensor_name=slope_name,
        scale_log2=-scalar * delta_factor,
        index_mode=slope_index.mode if slope_index is not None else None,
        index_divisor=slope_index.divisor if slope_index is not None else None,
    )


def _attention_relative_or_alibi_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    add = _attention_add_score_user(score_node)
    if add is None:
        return None
    add_node, bias_node = add
    alibi = _attention_alibi_bias_modifier(
        bias_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if alibi is not None:
        return add_node, alibi
    factor = _attention_index_delta_factor(
        bias_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if factor is None or math.isclose(factor, 0.0, abs_tol=1e-12):
        return None
    return add_node, AttentionScoreModifier(RELATIVE_BIAS_KIND, scale_log2=factor)


def _attention_doc_id_load(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[str, str, _AttentionBatchIndex] | None:
    from ..language import view_ops

    while (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is view_ops.subscript
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    if not isinstance(node, torch.fx.Node):
        return None
    tensor_name = _attention_load_host_tensor_name(node)
    if tensor_name is None:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        return None
    batch_idx, seq_idx = indices
    index_mode = _attention_collapsed_batch_index_mode(
        batch_idx,
        allow_mod=False,
        allow_floordiv=True,
    )
    if index_mode is None:
        return None
    if not isinstance(seq_idx, torch.fx.Node):
        return None
    if _attention_is_q_block_index(seq_idx, q_placeholder):
        return tensor_name, "q", index_mode
    if kv_block_id is not None and _attention_is_block_symnode(
        seq_idx, int(kv_block_id)
    ):
        return tensor_name, "kv", index_mode
    return None


def _attention_bool_terms(node: object, targets: frozenset[object]) -> list[object]:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
        and len(node.args) >= 2
    ):
        return _attention_bool_terms(node.args[0], targets) + _attention_bool_terms(
            node.args[1], targets
        )
    return [node]


_ATTENTION_AND_TARGETS: frozenset[object] = frozenset(
    {
        operator.and_,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.logical_and.default,
    }
)
_ATTENTION_OR_TARGETS: frozenset[object] = frozenset(
    {
        operator.or_,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.logical_or.default,
    }
)


def _attention_is_causal_predicate(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node.target is torch.ops.aten.ge.Tensor and len(node.args) >= 2:
        lhs, rhs = node.args[:2]
        if _attention_is_q_tile_index(
            lhs, q_placeholder
        ) and _attention_is_kv_tile_index(rhs, kv_block_id):
            return True
    if node.target in (torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar):
        if len(node.args) < 2:
            return False
        delta = _attention_index_delta_factor(
            node.args[0],
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        bound = _attention_scalar_value(node.args[1])
        return (
            delta is not None
            and bound is not None
            and math.isclose(delta, 1.0, rel_tol=1e-6, abs_tol=1e-7)
            and math.isclose(bound, 0.0, abs_tol=1e-7)
        )
    return False


def _attention_sliding_window_bound(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> int | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar):
        return None
    if len(node.args) < 2:
        return None
    delta = _attention_index_delta_factor(
        node.args[0],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    bound = _attention_scalar_value(node.args[1])
    if (
        delta is None
        or bound is None
        or not math.isclose(delta, 1.0, rel_tol=1e-6, abs_tol=1e-7)
    ):
        return None
    window = int(bound)
    if not math.isclose(bound, float(window), abs_tol=1e-7) or window < 0:
        return None
    return window


def _attention_prefix_length(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> int | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    if node.target not in (torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar):
        return None
    if len(node.args) < 2:
        return None
    if not _attention_is_kv_tile_index(node.args[0], kv_block_id):
        return None
    bound = _attention_scalar_value(node.args[1])
    if bound is None:
        return None
    prefix = int(bound)
    if not math.isclose(bound, float(prefix), abs_tol=1e-7) or prefix < 0:
        return None
    return prefix


def _attention_document_mask_name(
    node: object,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[str, _AttentionBatchIndex] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not torch.ops.aten.eq.Tensor
        or len(node.args) < 2
    ):
        return None
    lhs = _attention_doc_id_load(
        node.args[0],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    rhs = _attention_doc_id_load(
        node.args[1],
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if lhs is None or rhs is None or lhs[0] != rhs[0] or lhs[2] != rhs[2]:
        return None
    return (lhs[0], lhs[2]) if {lhs[1], rhs[1]} == {"q", "kv"} else None


def _attention_mask_modifier_from_predicate(
    pred: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> AttentionScoreModifier | None:
    if _attention_is_causal_predicate(
        pred,
        q_placeholder,
        kv_block_id=kv_block_id,
    ):
        return AttentionScoreModifier(CAUSAL_MASK_KIND)

    or_terms = _attention_bool_terms(pred, _ATTENTION_OR_TARGETS)
    if len(or_terms) == 2:
        prefix = None
        has_causal = False
        for term in or_terms:
            if _attention_is_causal_predicate(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            ):
                has_causal = True
                continue
            term_prefix = _attention_prefix_length(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_prefix is not None:
                prefix = term_prefix
        if prefix is not None and has_causal:
            return AttentionScoreModifier(PREFIX_LM_MASK_KIND, prefix_length=prefix)

    and_terms = _attention_bool_terms(pred, _ATTENTION_AND_TARGETS)
    if len(and_terms) >= 2:
        has_causal = False
        window = None
        document_name = None
        document_index: _AttentionBatchIndex | None = None
        for term in and_terms:
            if _attention_is_causal_predicate(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            ):
                has_causal = True
                continue
            term_window = _attention_sliding_window_bound(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_window is not None:
                if window is not None:
                    return None
                window = term_window
                continue
            term_document = _attention_document_mask_name(
                term,
                q_placeholder,
                kv_block_id=kv_block_id,
            )
            if term_document is not None:
                if document_name is not None:
                    return None
                document_name, document_index = term_document
                continue
            return None
        if has_causal and window is not None and document_name is None:
            return AttentionScoreModifier(
                SLIDING_WINDOW_MASK_KIND,
                window_size=window,
            )
        if has_causal and document_name is not None and window is None:
            return AttentionScoreModifier(
                DOCUMENT_MASK_KIND,
                tensor_name=document_name,
                index_mode=document_index.mode if document_index is not None else None,
                index_divisor=document_index.divisor
                if document_index is not None
                else None,
            )
    return None


def _attention_mask_score_node(
    score_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.where.self
    ]
    if len(users) != 1 or set(score_node.users) != {users[0]}:
        return None
    where_node = users[0]
    if len(where_node.args) < 3 or where_node.args[1] is not score_node:
        return None
    if not _attention_is_negative_infinity(where_node.args[2]):
        return None
    pred = where_node.args[0]
    if not isinstance(pred, torch.fx.Node):
        return None
    modifier = _attention_mask_modifier_from_predicate(
        pred,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if modifier is None:
        return None
    return where_node, modifier


def _attention_softcap_score_node(
    score_node: torch.fx.Node,
) -> tuple[torch.fx.Node, AttentionScoreModifier] | None:
    div_users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.div.Tensor
    ]
    if len(div_users) != 1 or set(score_node.users) != {div_users[0]}:
        return None
    div_node = div_users[0]
    if len(div_node.args) < 2 or div_node.args[0] is not score_node:
        return None
    softcap = _attention_scalar_value(div_node.args[1])
    if softcap is None or softcap <= 0.0:
        return None
    tanh_users = [
        user
        for user in div_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.tanh.default
    ]
    if len(tanh_users) != 1 or set(div_node.users) != {tanh_users[0]}:
        return None
    tanh_node = tanh_users[0]
    mul_users = [
        user
        for user in tanh_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.mul.Tensor
    ]
    if len(mul_users) != 1 or set(tanh_node.users) != {mul_users[0]}:
        return None
    mul_node = mul_users[0]
    if len(mul_node.args) < 2:
        return None
    if mul_node.args[0] is tanh_node:
        cap_arg = mul_node.args[1]
    elif mul_node.args[1] is tanh_node:
        cap_arg = mul_node.args[0]
    else:
        return None
    if not math.isclose(
        _attention_scalar_value(cap_arg) or float("nan"),
        softcap,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        return None
    return mul_node, AttentionScoreModifier(SOFTCAP_KIND, value_log2=softcap)


def _attention_score_modifiers(
    qk_node: torch.fx.Node,
    q_placeholder: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, tuple[AttentionScoreModifier, ...]] | None:
    score_node = qk_node
    modifiers: list[AttentionScoreModifier] = []
    while True:
        bias = _attention_tensor_bias_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if bias is not None:
            score_node, modifier = bias
            modifiers.append(modifier)
            continue
        index_bias = _attention_relative_or_alibi_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if index_bias is not None:
            score_node, modifier = index_bias
            modifiers.append(modifier)
            continue
        softcap = _attention_softcap_score_node(score_node)
        if softcap is not None:
            score_node, modifier = softcap
            modifiers.append(modifier)
            continue
        mask = _attention_mask_score_node(
            score_node,
            q_placeholder,
            kv_block_id=kv_block_id,
        )
        if mask is not None:
            score_node, modifier = mask
            modifiers.append(modifier)
            continue
        return score_node, tuple(modifiers)


def _attention_canonical_kv_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    from ..language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.load:
        return None
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    if not isinstance(indices[0], torch.fx.Node) or not isinstance(
        indices[1], torch.fx.Node
    ):
        return None
    if not _attention_is_full_slice(indices[2]):
        return None
    if not _attention_is_inner_batch_index(indices[0]):
        return None
    if kv_block_id is not None and not _attention_is_block_symnode(
        indices[1], int(kv_block_id)
    ):
        return None
    return indices[0], indices[1]


def _attention_k_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if node.op != "call_function":
        return None
    if node.target is torch.ops.aten.permute.default:
        if len(node.args) < 2 or node.args[1] != [0, 2, 1]:
            return None
        source = node.args[0]
    elif node.target is torch.ops.aten.transpose.int:
        if len(node.args) < 3 or node.args[1] != 1 or node.args[2] != 2:
            return None
        source = node.args[0]
    else:
        return None
    if not isinstance(source, torch.fx.Node):
        return None
    return _attention_canonical_kv_load_indices(source, kv_block_id=kv_block_id)


def _attention_v_load_indices(
    node: torch.fx.Node,
    *,
    kv_block_id: int | None,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    return _attention_canonical_kv_load_indices(node, kv_block_id=kv_block_id)


def _attention_exact_online_softmax_chain(
    score_node: torch.fx.Node, pv_node: torch.fx.Node
) -> bool:
    from ..language import view_ops
    from ..language._tracing_ops import _mask_to

    def broadcast_row_vector_source(node: torch.fx.Node) -> torch.fx.Node | None:
        if node.op != "call_function" or node.target is not view_ops.subscript:
            return None
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            return None
        indices = node.args[1] if len(node.args) > 1 else None
        if not isinstance(indices, (list, tuple)) or len(indices) != 3:
            return None
        if (
            not _attention_is_full_slice(indices[0])
            or not _attention_is_full_slice(indices[1])
            or indices[2] is not None
        ):
            return None
        return node.args[0]

    def is_broadcast_row_vector(node: torch.fx.Node, source: torch.fx.Node) -> bool:
        return broadcast_row_vector_source(node) is source

    def is_squeeze_last_dim(node: torch.fx.Node, source: torch.fx.Node) -> bool:
        dim = node.args[1] if len(node.args) >= 2 else None
        return (
            node.op == "call_function"
            and node.target is torch.ops.aten.squeeze.dim
            and len(node.args) >= 2
            and node.args[0] is source
            and isinstance(dim, int)
            and dim == -1
        )

    def binary_node_has_args(
        node: torch.fx.Node,
        target: object,
        lhs: torch.fx.Node,
        rhs: torch.fx.Node,
        *,
        ordered: bool,
    ) -> bool:
        if node.op != "call_function" or node.target is not target:
            return False
        if len(node.args) < 2:
            return False
        if ordered:
            return node.args[0] is lhs and node.args[1] is rhs
        return (node.args[0] is lhs and node.args[1] is rhs) or (
            node.args[0] is rhs and node.args[1] is lhs
        )

    def unwrap_new_var(node: object) -> object:
        if isinstance(node, torch.fx.Node):
            return _attention_new_var_source(node) or node
        return node

    def loop_body_returns(
        max_node: torch.fx.Node,
        l_update: torch.fx.Node,
        acc_update: torch.fx.Node,
    ) -> bool:
        for node in score_node.graph.nodes:
            if node.op != "output":
                continue
            if len(node.args) != 1 or not isinstance(node.args[0], (list, tuple)):
                return False
            outputs = node.args[0]
            if len(outputs) != 3:
                return False
            return (
                unwrap_new_var(outputs[0]) is max_node
                and unwrap_new_var(outputs[1]) is l_update
                and unwrap_new_var(outputs[2]) is acc_update
            )
        return False

    prev_m_placeholder = _attention_loop_carried_arg(score_node.graph, 1)
    prev_l_placeholder = _attention_loop_carried_arg(score_node.graph, 2)
    prev_acc_placeholder = _attention_loop_carried_arg(score_node.graph, 3)
    if (
        prev_m_placeholder is None
        or prev_l_placeholder is None
        or prev_acc_placeholder is None
    ):
        return False

    qk_mask_users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is _mask_to
    ]
    qk_sub_users = [
        user
        for user in score_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.sub.Tensor
    ]
    if len(qk_mask_users) != 1 or len(qk_sub_users) != 1:
        return False
    if set(score_node.users) != {qk_mask_users[0], qk_sub_users[0]}:
        return False

    qk_mask = qk_mask_users[0]
    if len(qk_mask.args) < 2 or qk_mask.args[0] is not score_node:
        return False
    if qk_mask.args[1] != float("-inf"):
        return False
    amax_users = [
        user
        for user in qk_mask.users
        if user.op == "call_function" and user.target is torch.ops.aten.amax.default
    ]
    if len(amax_users) != 1 or set(qk_mask.users) != {amax_users[0]}:
        return False
    amax_node = amax_users[0]
    if len(amax_node.args) < 2 or amax_node.args[1] != [-1]:
        return False
    amax_keepdim = len(amax_node.args) >= 3 and bool(amax_node.args[2])
    maximum_users = [
        user
        for user in amax_node.users
        if user.op == "call_function" and user.target is torch.ops.aten.maximum.default
    ]
    if len(maximum_users) != 1:
        return False
    max_node = maximum_users[0]
    if len(max_node.args) < 2:
        return False
    if max_node.args[0] is amax_node and isinstance(max_node.args[1], torch.fx.Node):
        prev_m_arg = max_node.args[1]
    elif max_node.args[1] is amax_node and isinstance(max_node.args[0], torch.fx.Node):
        prev_m_arg = max_node.args[0]
    else:
        return False
    if amax_keepdim:
        prev_m = broadcast_row_vector_source(prev_m_arg)
        if prev_m is None:
            return False
        max_state_users = [
            user for user in max_node.users if is_squeeze_last_dim(user, max_node)
        ]
        if len(max_state_users) != 1:
            return False
        max_state_node = max_state_users[0]
    else:
        prev_m = prev_m_arg
        max_state_node = max_node
    if not _attention_is_loop_carried_value(prev_m, prev_m_placeholder):
        return False

    centered = qk_sub_users[0]
    if len(centered.args) < 2 or centered.args[0] is not score_node:
        return False
    max_view = centered.args[1]
    if not isinstance(max_view, torch.fx.Node):
        return False
    if amax_keepdim:
        if max_view is not max_node:
            return False
    else:
        if not is_broadcast_row_vector(max_view, max_node):
            return False

    p_users = [
        user
        for user in centered.users
        if user.op == "call_function"
        and user.target in (torch.ops.aten.exp.default, torch.ops.aten.exp2.default)
    ]
    if len(p_users) != 1 or set(centered.users) != {p_users[0]}:
        return False
    p_node = p_users[0]

    p_mask_users = [
        user
        for user in p_node.users
        if user.op == "call_function"
        and user.target is _mask_to
        and len(user.args) >= 2
        and user.args[0] is p_node
        and user.args[1] == 0
    ]
    p_sum_users = [
        user
        for p_mask in p_mask_users
        for user in p_mask.users
        if user.op == "call_function"
        and user.target is torch.ops.aten.sum.dim_IntList
        and len(user.args) >= 2
        and user.args[1] == [-1]
    ]
    if len(p_sum_users) != 1:
        return False
    l_ij = p_sum_users[0]

    alpha_sub_users = [
        user
        for user in max_state_node.users
        if binary_node_has_args(
            user,
            torch.ops.aten.sub.Tensor,
            prev_m,
            max_state_node,
            ordered=True,
        )
    ]
    if len(alpha_sub_users) != 1:
        return False
    alpha_sub = alpha_sub_users[0]
    alpha_users = [
        user
        for user in alpha_sub.users
        if user.op == "call_function" and user.target is p_node.target
    ]
    if len(alpha_users) != 1 or set(alpha_sub.users) != {alpha_users[0]}:
        return False
    alpha = alpha_users[0]

    l_mul_candidates = [
        user
        for user in alpha.users
        if user.op == "call_function"
        and user.target is torch.ops.aten.mul.Tensor
        and len(user.args) >= 2
        and alpha in user.args[:2]
        and any(
            isinstance(arg, torch.fx.Node)
            and _attention_is_loop_carried_value(arg, prev_l_placeholder)
            for arg in user.args[:2]
        )
    ]
    l_update_candidates = [
        user
        for l_mul in l_mul_candidates
        for user in l_mul.users
        if binary_node_has_args(
            user,
            torch.ops.aten.add.Tensor,
            l_mul,
            l_ij,
            ordered=False,
        )
    ]
    if len(l_update_candidates) != 1:
        return False
    l_update = l_update_candidates[0]

    alpha_views = [user for user in alpha.users if is_broadcast_row_vector(user, alpha)]
    if len(alpha_views) != 1:
        return False
    alpha_view = alpha_views[0]
    acc_rescale_candidates = [
        user
        for user in alpha_view.users
        if user.op == "call_function"
        and user.target is torch.ops.aten.mul.Tensor
        and len(user.args) >= 2
        and alpha_view in user.args[:2]
        and any(
            isinstance(arg, torch.fx.Node)
            and _attention_is_loop_carried_value(arg, prev_acc_placeholder)
            for arg in user.args[:2]
        )
    ]
    if len(acc_rescale_candidates) != 1:
        return False
    acc_rescaled = acc_rescale_candidates[0]

    if len(pv_node.args) < 3 or not isinstance(pv_node.args[1], torch.fx.Node):
        return False
    if pv_node.args[0] is not acc_rescaled:
        return False
    if _attention_pv_p_arg_base(pv_node.args[1]) is not p_node:
        return False
    return loop_body_returns(max_state_node, l_update, pv_node)


def _attention_online_softmax_exp_base(score_node: torch.fx.Node) -> str | None:
    for centered in score_node.users:
        if (
            centered.op != "call_function"
            or centered.target is not torch.ops.aten.sub.Tensor
            or len(centered.args) < 2
            or centered.args[0] is not score_node
        ):
            continue
        for user in centered.users:
            if user.op != "call_function":
                continue
            if user.target is torch.ops.aten.exp2.default:
                return "exp2"
            if user.target is torch.ops.aten.exp.default:
                return "exp"
    return None


def _attention_softmax_pattern_head_dim(
    graph: torch.fx.Graph,
    *,
    kv_block_id: int | None = None,
) -> AttentionSoftmaxPattern | None:
    """Config-independent online-softmax attention detector.

    Returns the head_dim (in {64, 128}) and whether the canonical causal mask is
    present when ``graph`` is a flash body -- a QK ``bmm`` + PV ``baddbmm``
    feeding an ``amax``/``exp2``/``sum`` online softmax with either no score
    masking or exactly ``where(tile_m.index >= tile_n.index, qk, -inf)``. Shared
    by ``_attention_loop_shape`` (the codegen detector) and the autotuner flash
    seed heuristic so the two never diverge on what counts as a flash kernel.
    """
    qk_nodes: list[torch.fx.Node] = []
    pv_nodes: list[torch.fx.Node] = []
    has_amax = has_exp = has_sum = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if target is torch.ops.aten.bmm.dtype:
            qk_nodes.append(node)
        elif target is torch.ops.aten.baddbmm.default:
            pv_nodes.append(node)
        elif target is torch.ops.aten.amax.default:
            has_amax = True
        elif target in (torch.ops.aten.exp.default, torch.ops.aten.exp2.default):
            has_exp = True
        elif target is torch.ops.aten.sum.dim_IntList:
            has_sum = True
    if len(qk_nodes) != 1 or len(pv_nodes) != 1:
        return None
    qk_node = qk_nodes[0]
    pv_node = pv_nodes[0]
    if not (has_amax and has_exp and has_sum):
        return None
    qk_val = qk_node.meta.get("val")
    pv_val = pv_node.meta.get("val")
    if not isinstance(qk_val, torch.Tensor) or not isinstance(pv_val, torch.Tensor):
        return None
    # QK score block: (batch, tile_m, tile_n); PV output: (batch, tile_m, head_dim).
    if qk_val.ndim != 3 or pv_val.ndim != 3:
        return None
    if qk_val.dtype != torch.float32 or pv_val.dtype != torch.float32:
        return None
    # The emitted kernel supports 16-bit floating Q/K/V operands. Checked via FX
    # node metadata (available at detection time, unlike the device-function
    # arguments).
    operand_nodes = [qk_node.args[0], qk_node.args[1]]
    if len(pv_node.args) > 2:
        operand_nodes.append(pv_node.args[2])
    operand_dtype: torch.dtype | None = None
    for arg in operand_nodes:
        if not isinstance(arg, torch.fx.Node):
            return None
        operand_val = arg.meta.get("val")
        if not isinstance(operand_val, torch.Tensor) or operand_val.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return None
        if operand_dtype is None:
            operand_dtype = operand_val.dtype
        elif operand_val.dtype != operand_dtype:
            return None
    head_dim = pv_val.shape[2]
    if not isinstance(head_dim, int) or head_dim not in (64, 128):
        return None
    if not isinstance(qk_node.args[1], torch.fx.Node):
        return None
    k_indices = _attention_k_load_indices(qk_node.args[1], kv_block_id=kv_block_id)
    if k_indices is None:
        return None
    if len(pv_node.args) < 3 or not isinstance(pv_node.args[2], torch.fx.Node):
        return None
    v_indices = _attention_v_load_indices(pv_node.args[2], kv_block_id=kv_block_id)
    if v_indices is None:
        return None
    if k_indices[0] is not v_indices[0] or k_indices[1] is not v_indices[1]:
        return None
    q_placeholder = _attention_loop_carried_arg(graph, 0)
    if q_placeholder is None:
        return None
    if not isinstance(qk_node.args[0], torch.fx.Node):
        return None
    q_scale = _attention_arg_scaled_q_factor(qk_node.args[0], q_placeholder)
    if q_scale is None:
        return None
    modifiers_result = _attention_score_modifiers(
        qk_node,
        q_placeholder,
        kv_block_id=kv_block_id,
    )
    if modifiers_result is None:
        return None
    score_node, modifiers = modifiers_result
    if not _attention_exact_online_softmax_chain(score_node, pv_node):
        return None
    exp_base = _attention_online_softmax_exp_base(score_node)
    if exp_base == "exp2":
        expected_scale = math.log2(math.e) / math.sqrt(head_dim)
        qk_scale_log2 = expected_scale
        bias_scale_log2 = 1.0
        lse_scale = 1.0
    elif exp_base == "exp":
        expected_scale = 1.0 / math.sqrt(head_dim)
        qk_scale_log2 = math.log2(math.e) / math.sqrt(head_dim)
        bias_scale_log2 = math.log2(math.e)
        lse_scale = math.log(2.0)
    else:
        return None
    if not math.isclose(q_scale, expected_scale, rel_tol=1e-5, abs_tol=1e-7):
        return None
    additive_modifier_kinds = {
        TENSOR_BIAS_KIND,
        RELATIVE_BIAS_KIND,
        ALIBI_BIAS_KIND,
    }
    scaled_modifiers = tuple(
        dataclasses.replace(
            modifier,
            scale_log2=modifier.scale_log2 * bias_scale_log2,
        )
        if modifier.kind in additive_modifier_kinds
        else dataclasses.replace(
            modifier,
            value_log2=(
                None
                if modifier.value_log2 is None
                else modifier.value_log2 * bias_scale_log2
            ),
        )
        if modifier.kind == SOFTCAP_KIND
        else modifier
        for modifier in modifiers
    )
    score_plan = AttentionScorePlan(
        head_dim=head_dim,
        qk_scale_log2=qk_scale_log2,
        lse_scale=lse_scale,
        modifiers=scaled_modifiers,
    )
    if not score_plan.has_lowering():
        return None
    return AttentionSoftmaxPattern(score_plan=score_plan)


def detect_flash_search_surface(device_ir: DeviceIR) -> FlashSearchSurface | None:
    """Config-independent flash detector for the autotune search surface.

    Returns the flash head_dim, KV tile count, and block-size targets when
    ``device_ir`` is inside the same static shape envelope as the fused tcgen05
    path: square fp16/bf16 self-attention, concrete ``seq % 128 == 0``, and reachable
    ``block_sizes=[1, 128, 128]`` for ``(tile_b, tile_m, tile_n)``. Keeping this
    strict prevents the autotuner from benchmarking configs that can only fall
    back to the scalar path after the flash knobs have been added.
    """
    from ..autotuner.config_fragment import BlockSizeFragment
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo

    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return None
    # Attention binds to a 2-axis root grid (tile_b, tile_m) plus one inner
    # tile_n device loop.
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 2:
        return None
    env = CompileEnvironment.current()
    for graph_info in device_ir.graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = graph_info.block_ids
        if len(block_ids) != 1 or any(bid in root_grid_ids for bid in block_ids):
            continue
        pattern = _attention_softmax_pattern_head_dim(
            graph_info.graph,
            kv_block_id=block_ids[0],
        )
        if pattern is None:
            continue
        from .cute.cute_flash import flash_attention_graph_lse_plan_valid_from_graphs
        from .cute.cute_flash import (
            flash_attention_graph_small_biased_candidate_from_graphs,
        )

        if not flash_attention_graph_lse_plan_valid_from_graphs(
            device_ir.graphs,
            root_block_ids=root_grid_ids,
            kv_block_id=block_ids[0],
            score_plan=pattern.score_plan,
        ):
            continue
        small_biased_candidate = (
            flash_attention_graph_small_biased_candidate_from_graphs(
                device_ir.graphs,
                root_block_ids=root_grid_ids,
                kv_block_id=block_ids[0],
                score_plan=pattern.score_plan,
            )
        )
        q_seq = env.block_sizes[root_grid_ids[1]].size
        kv_seq = env.block_sizes[block_ids[0]].size
        if not (isinstance(q_seq, int) and isinstance(kv_seq, int) and q_seq == kv_seq):
            continue
        if q_seq % 128 != 0:
            continue
        block_size_targets = {
            root_grid_ids[0]: 1,
            root_grid_ids[1]: 128,
            block_ids[0]: 128,
        }
        if set(env.config_spec.block_sizes.valid_block_ids()) != set(
            block_size_targets
        ):
            continue
        reachable = True
        for block_id, target in block_size_targets.items():
            try:
                block_spec = env.config_spec.block_sizes.block_id_lookup(block_id)
            except KeyError:
                reachable = False
                break
            fragment = block_spec._fragment(env.config_spec)
            assert isinstance(fragment, BlockSizeFragment)
            if not fragment.low <= target <= fragment.high:
                reachable = False
                break
        if not reachable:
            continue
        return FlashSearchSurface(
            head_dim=pattern.head_dim,
            num_kv=(kv_seq + 127) // 128,
            block_size_targets=block_size_targets,
            is_causal=pattern.is_causal,
            has_kv_tile_pruning=pattern.score_plan.has_kv_tile_pruning,
            requires_ws_overlap=pattern.score_plan.requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
        )
    return None


class _SpecializedMmaPlan(NamedTuple):
    impl: str
    m_block_id: int
    n_block_id: int


def _kernel_specialized_mma_plan(
    fn: DeviceFunction,
    *,
    config: Config,
) -> _SpecializedMmaPlan | None:
    from .compile_environment import CompileEnvironment
    from .cute.cute_mma import _choose_mma_impl
    from .cute.cute_mma import _mma_tiles_are_static_full
    from .cute.cute_mma import analyze_cute_mma_node
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction

    env = CompileEnvironment.current()
    grid_ids = {
        bid for ids in HostFunction.current().device_ir.grid_block_ids for bid in ids
    }
    seen_block_ids: set[tuple[int, ...]] = set()
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = tuple(graph_info.block_ids)
        if block_ids in seen_block_ids or not any(
            bid not in grid_ids for bid in block_ids
        ):
            continue
        seen_block_ids.add(block_ids)
        block_sizes = [env.block_sizes[bid].from_config(config) for bid in block_ids]
        if len(block_sizes) != 1 or not isinstance(block_sizes[0], int):
            continue
        bk = block_sizes[0]
        for node in graph_info.graph.nodes:
            candidate = analyze_cute_mma_node(node)
            if (
                candidate is None
                or candidate.requires_accumulator_seed
                or candidate.operands.k_block_id != block_ids[0]
            ):
                continue
            root_mn_block_ids = _specialized_mma_root_mn_block_ids(candidate, config)
            if root_mn_block_ids is None:
                continue
            bm = env.block_sizes[root_mn_block_ids[0]].from_config(config)
            bn = env.block_sizes[root_mn_block_ids[1]].from_config(config)
            if not isinstance(bm, int) or not isinstance(bn, int):
                continue
            if (
                candidate.operands.has_leading_passthrough
                and not _mma_tiles_are_static_full(
                    candidate.operands, bm=bm, bn=bn, bk=bk
                )
            ):
                continue
            lhs_val = candidate.operands.lhs.source_fake
            mma_impl = _choose_mma_impl(
                lhs_val.dtype,
                bm=bm,
                bn=bn,
                bk=bk,
                config=config,
                input_device=lhs_val.device,
            )
            if mma_impl != "universal":
                return _SpecializedMmaPlan(mma_impl, *root_mn_block_ids)
    return None


def _kernel_specialized_mma_impl(
    fn: DeviceFunction,
    *,
    config: Config,
) -> str | None:
    plan = _kernel_specialized_mma_plan(fn, config=config)
    return None if plan is None else plan.impl


def _loop_contains_matmul(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    from ..language._decorators import is_api_func
    from .device_ir import RootGraphInfo
    from .host_function import HostFunction

    matmul_targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    device_ir = HostFunction.current().device_ir
    graph_by_id = {
        graph_info.graph_id: graph_info
        for graph_info in fn.codegen.codegen_graphs
        if hasattr(graph_info, "graph")
    }

    def graph_contains_matmul(graph: object) -> bool:
        if not isinstance(graph, torch.fx.Graph):
            return False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in matmul_targets:
                return True
            if is_api_func(node.target):
                name = getattr(node.target, "__name__", "")
                if name == "dot":
                    return True
                if name in {"_for_loop", "_for_loop_step"}:
                    graph_id = node.args[0] if node.args else None
                    if isinstance(graph_id, int):
                        nested = graph_by_id.get(graph_id)
                        if nested is not None and graph_contains_matmul(nested.graph):
                            return True
        return False

    def graph_matches_loop(graph_info: object) -> bool:
        if getattr(graph_info, "block_ids", None) == block_ids:
            return True
        if not isinstance(graph_info, RootGraphInfo):
            return False
        phase_index = graph_info.phase_index
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    for graph_info in fn.codegen.codegen_graphs:
        if not graph_matches_loop(graph_info):
            continue
        if graph_contains_matmul(graph_info.graph):
            return True
    return False


def _active_loop_block_ids(fn: DeviceFunction) -> set[int]:
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    active: set[int] = {
        block_id for block_ids in device_ir.grid_block_ids for block_id in block_ids
    }
    for graph_info in fn.codegen.codegen_graphs:
        block_ids = getattr(graph_info, "block_ids", None)
        if block_ids is None:
            continue
        active.update(block_ids)
    return active


# The backend subclasses live in per-backend modules under helion/_compiler/<backend>/backend.py.
# Re-import them here so `from helion._compiler.backend import <Backend>` keeps resolving and the
# classes register with the same timing as when they lived in this file -- no behavior change.
from .cute.backend import CuteBackend  # noqa: E402, F401
from .metal.backend import MetalBackend  # noqa: E402, F401
from .pallas.backend import PallasBackend  # noqa: E402, F401
from .triton.backend import TileIRBackend  # noqa: E402, F401
from .triton.backend import TritonBackend  # noqa: E402, F401
