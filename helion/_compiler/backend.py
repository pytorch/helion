from __future__ import annotations

import abc
import ast
import base64
import contextlib
import dataclasses
import enum
import functools
import hashlib
from itertools import starmap
import logging
import math
import operator
import os
import re
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
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
from .cute.tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.ops_handler import OpsHandler

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..autotuner.config_priors import ValuePrior
    from ..autotuner.config_spec import ConfigSpec
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.settings import DotPrecision
    from .device_function import Argument
    from .device_function import DeviceFunction
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .host_function import HostFunction
    from .pallas.compact_worklist import CompactWorklistPlan
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


@functools.cache
def _triton_jit_supports_do_not_specialize() -> bool:
    try:
        import inspect

        import triton
    except ImportError:
        return False

    params = inspect.signature(triton.jit).parameters
    return "do_not_specialize" in params and "do_not_specialize_on_alignment" in params


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
        from .device_function import texpr

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


class TritonBackend(Backend):
    """Triton code generation backend."""

    @property
    def name(self) -> str:
        return "triton"

    @property
    def experimental(self) -> bool:
        return False

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        from .device_function import TensorArg

        # Bind fp4x2 storage as uint8; Triton has no pointer type for the shell dtype.
        if (
            isinstance(arg, TensorArg)
            and arg.fake_value.dtype is torch.float4_e2m1fn_x2
        ):
            return f"{host_str}.view(torch.uint8)"
        return host_str

    def supports_config_key(self, key: str) -> bool:
        if key in ("load_cache_modifiers", "store_cache_modifiers"):
            return True
        if key == "waves_per_eu":
            from .._compat import is_hip

            return is_hip()
        if key == "matrix_instr_nonkdim":
            from .._compat import supports_amd_cdna_tunables

            return supports_amd_cdna_tunables()

        from .._compat import get_mtia_tunable_fragments
        from .._compat import supports_mtia_tunables

        if key in get_mtia_tunable_fragments():
            return supports_mtia_tunables()
        return super().supports_config_key(key)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from .._compat import get_mtia_tunable_fragments
        from .._compat import is_hip
        from .._compat import supports_amd_cdna_tunables
        from .._compat import supports_mtia_tunables
        from ..autotuner.config_fragment import EnumFragment

        if not is_hip() and not supports_mtia_tunables():
            return {}
        fragments: dict[str, ConfigSpecFragment] = {}
        if is_hip():
            fragments["waves_per_eu"] = EnumFragment(choices=(1, 2, 3, 4))
            if supports_amd_cdna_tunables():
                fragments["matrix_instr_nonkdim"] = EnumFragment(choices=(0, 16))

        if supports_mtia_tunables():
            fragments.update(get_mtia_tunable_fragments())

        return fragments

    def setup_compile_cache_dir(self, device_index: int) -> None:
        if "TRITON_CACHE_DIR" not in os.environ:
            from ..autotuner.local_cache import helion_triton_cache_dir

            triton_dir = helion_triton_cache_dir(device_index)
            os.environ["TRITON_CACHE_DIR"] = triton_dir
            log.debug("Set TRITON_CACHE_DIR=%s", triton_dir)

    def make_ephemeral_cache(
        self,
    ) -> contextlib.AbstractContextManager[None] | None:
        # HELION_KEEP_TRITON_CACHE is a deprecated alias kept for backward
        # compatibility; HELION_KEEP_CACHE is the canonical control.
        if (
            self.keep_compile_cache_requested()
            or os.environ.get("HELION_KEEP_TRITON_CACHE", "") == "1"
        ):
            return None
        return self._ephemeral_triton_cache()

    @contextlib.contextmanager
    def _ephemeral_triton_cache(self) -> Generator[None, None, None]:
        """Redirect Triton cache to a temporary dir during autotuning.

        All candidate compilations write to an ephemeral directory that is
        deleted on exit.  The winning config is recompiled afterward into the
        real cache by the caller.
        """
        saved = os.environ.get("TRITON_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="helion_autotune_") as ephemeral:
            os.environ["TRITON_CACHE_DIR"] = ephemeral
            log.debug("Ephemeral Triton cache: %s", ephemeral)
            try:
                yield
            finally:
                if saved is not None:
                    os.environ["TRITON_CACHE_DIR"] = saved
                else:
                    os.environ.pop("TRITON_CACHE_DIR", None)

    def finalize_ephemeral_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        from ..runtime.config import Config

        self._clear_triton_jit_cache(bound_kernel, config)
        evict = config
        if bound_kernel._compile_cache.pop(evict, None) is None:
            default = bound_kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            evict = Config(**(default.config | config.config))
            bound_kernel._compile_cache.pop(evict, None)
        bound_kernel._cache_path_map.pop(evict, None)

    def _clear_triton_jit_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        """Clear Triton's in-memory JIT cache for the compiled kernel.

        After autotuning in an ephemeral cache dir, device_caches on the
        JITFunction still holds the compiled binary.  Clearing it forces
        Triton to recompile (and write to TRITON_CACHE_DIR) on the next call.

        If the config was minimized by the autotuner, the lookup is retried
        with the full config (defaults merged back in).
        """
        from ..runtime.config import Config

        compiled_fn = bound_kernel._compile_cache.get(config)
        if compiled_fn is None:
            default = bound_kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            full_config = Config(**(default.config | config.config))
            compiled_fn = bound_kernel._compile_cache.get(full_config)
        if compiled_fn is None:
            return
        triton_jit_fn = compiled_fn.__globals__.get(
            f"_helion_{bound_kernel.kernel.name}"
        )
        if triton_jit_fn is not None and hasattr(triton_jit_fn, "device_caches"):
            triton_jit_fn.device_caches.clear()

    def compiled_cache_key(
        self, bound_kernel: BoundKernel[Any], compiled_fn: object
    ) -> str | None:
        # The jit_fn that - for helion - starts with _helion_
        triton_jit_fn = compiled_fn.__globals__.get(  # type: ignore[attr-defined]
            f"_helion_{bound_kernel.kernel.name}"
        )
        if triton_jit_fn is None:
            return None
        try:
            for cache_tuple in triton_jit_fn.device_caches.values():
                compiled_kernels = cache_tuple[0]
                for compiled_kernel in compiled_kernels.values():
                    h = getattr(compiled_kernel, "hash", None)
                    if h is not None:
                        return base64.b32encode(bytes.fromhex(h)).decode().rstrip("=")
        except (AttributeError, IndexError, TypeError, ValueError):
            # device_caches, cache-tuple layout, and CompiledKernel.hash are
            # Triton-internal details that may change across Triton versions
            # return None gracefully if this fails
            return None
        return None

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.utils import triton_type

        return triton_type(dtype)

    def acc_type(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.triton import triton_acc_type

        return triton_acc_type(dtype)

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"tl.cast({expr_str}, {dtype_str})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = {lid} * {block_size_var} + tl.arange(0, {block_size_var}).to({dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + tl.arange(0, ({block_size_var})).to({dtype})"

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            return f"tl.load({tensor_name})"
        return f"tl.load({tensor_name} + {index_expr})"

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"tl.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"tl.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"tl.zeros({shape}, {dtype})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"tl.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"tl.broadcast_to({expr}, {shape})"

    def maybe_reshape_reduction(
        self,
        expr: str,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        target_shape_expr: str,
    ) -> str:
        # Triton reductions over a 1D tile produce a scalar even when
        # keepdim=True makes the logical result shape [1]. tl.reshape() only
        # accepts block tensors here, so leave the scalar and let later ops
        # broadcast it.
        if not source_shape and math.prod(target_shape) == 1:
            return expr
        return self.reshape_expr(expr, target_shape_expr)

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        # Triton requires block shapes to be powers of 2. As of triton 3.8.0
        # (triton-lang/triton#10687, 2026-06), is_power_of_two(0) returns False,
        # so validate_block_shape rejects a length-0 tensor. Emit a length-1
        # index instead -- the accompanying ``index < 0`` mask is all-False, so
        # nothing is ever loaded/stored for the empty reduction dimension.
        return f"tl.zeros([1], {dtype})"

    def static_rdim_size(self, numel: int) -> int:
        # Pair with reduction_index_zero_expr: an empty (length-0) axis uses a
        # length-1 index, so its block extent must also be 1, not 0. Triton
        # rejects length-0 block shapes (is_power_of_two(0) is False), and a
        # 0-extent value would also mismatch the length-1 index shape in the
        # generated tl.store/tl.load. The all-False mask keeps it a no-op.
        return max(super().static_rdim_size(numel), 1)

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"triton.next_power_of_2({expr})"

    @property
    def function_decorator(self) -> str:
        return "triton.jit"

    def function_decorator_for_args(self, args: Sequence[Argument]) -> str:
        from .compile_environment import CompileEnvironment
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        # Default to Triton's own behavior: let Triton specialize on values and
        # alignment.  This enables vectorized loads (constexpr 1 for inner
        # strides, divisibility-by-16 hint for sizes) at the cost of an
        # occasional Triton recompile when a value crosses a specialization
        # boundary (e.g. size 1 -> 2, alignment changes).
        if not CompileEnvironment.current().settings.triton_do_not_specialize:
            return self.function_decorator

        do_not_specialize = [
            arg.name
            for arg in args
            if isinstance(arg, (TensorSizeArg, TensorStrideArg, SymbolArgument))
        ]
        if not do_not_specialize or not _triton_jit_supports_do_not_specialize():
            return self.function_decorator
        return (
            "triton.jit("
            f"do_not_specialize={do_not_specialize!r}, "
            f"do_not_specialize_on_alignment={do_not_specialize!r})"
        )

    @property
    def constexpr_type(self) -> str:
        return "tl.constexpr"

    @property
    def default_launcher_name(self) -> str:
        return "_default_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "operator": "import operator",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "triton": "import triton",
            "tl": "import triton.language as tl",
            "triton_helpers": "from torch._inductor.runtime import triton_helpers",
            "tl_math": "from torch._inductor.runtime.triton_helpers import math as tl_math",
            "libdevice": "from torch._inductor.runtime.triton_compat import libdevice",
            "_default_launcher": "from helion.runtime import default_launcher as _default_launcher",
            "fast_dividef": "from triton.language.extra.libdevice import fast_dividef",
            "fast_expf": "from triton.language.extra.libdevice import fast_expf",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        if index_dtype != "tl.int32":
            return f"tl.program_id({dim}).to({index_dtype})"
        return f"tl.program_id({dim})"

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        if is_device:
            return f"tl.cdiv({numel}, {block_size})"
        return f"triton.cdiv({numel}, {block_size})"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from .triton.overrides import HelionTritonOverrides

        return HelionTritonOverrides()

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return f"{offset_var} + tl.zeros([1], {dtype})"
        return f"({offset_var} + tl.arange(0, ({block_size_var}))).to({dtype})"

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        if reduction_type in {"sum", "max", "min"}:
            return f"tl.{reduction_type}({input_name}, {dim})"
        if reduction_type == "prod":
            return f"triton_helpers.prod({input_name}, {dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

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
    ) -> str:
        helper = "max" if reduction_type == "argmax" else "min"
        return (
            f"triton_helpers.{helper}_with_index("
            f"{input_name}, {index_value}, {dim})[1].to({self.dtype_str(output_dtype)})"
        )

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        helper = "maximum" if reduction_type == "argmax" else "minimum"
        return [
            (
                f"{acc}, {acc_index} = "
                f"triton_helpers.{helper}_with_index({acc}, {acc_index}, {value}, {index})"
            )
        ]

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        return (
            f"tl.full([{', '.join(shape_dims)}], {value_expr}, {self.dtype_str(dtype)})"
        )

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from .._compat import supports_maxnreg

        # Workaround for triton bug: warp_specialize requires at least 4 warps
        # See: https://github.com/triton-lang/triton/issues/7354
        num_warps = config.num_warps
        if any(config.range_warp_specializes):
            num_warps = max(4, num_warps)

        args = [
            f"num_warps={num_warps}",
            f"num_stages={config.num_stages}",
            *(["launch_cooperative_grid=True"] if has_barrier else []),
        ] + [
            f"{x.removeprefix('_triton_config_')}={config[x]}"
            for x in config
            if x.startswith("_triton_config_")
        ]

        from ..autotuner.config_spec import _get_backend_tunable_keys

        for key in _get_backend_tunable_keys():
            if key in config:
                args.append(f"{key}={config[key]!r}")

        if "maxnreg" in config and config["maxnreg"] is not None and supports_maxnreg():
            args.append(f"maxnreg={config['maxnreg']}")

        advanced_controls_file = config.advanced_controls_file
        if advanced_controls_file:
            ptx_option = f"--apply-controls {advanced_controls_file}"
            args.append(f"ptx_options={ptx_option!r}")

        return args

    def grid_barrier_stmt(self, sem_arg: str) -> str:
        return f"triton_helpers.x_grid_barrier({sem_arg})"

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
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        return frozenset({"grid", "warmup", "num_warps", "num_stages"})


class TileIRBackend(TritonBackend):
    """TileIR code generation backend (extends Triton)."""

    @property
    def name(self) -> str:
        return "tileir"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def supports_config_key(self, key: str) -> bool:
        # Override TritonBackend/Backend rejections for tileir-specific tunables
        if key in {"num_ctas", "occupancy"}:
            return True
        return super().supports_config_key(key)

    def supports_block_ptr_indexing(self) -> bool:
        return False

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from ..autotuner.config_fragment import PowerOfTwoFragment

        return {
            **super().tunable_fragments(),
            "num_ctas": PowerOfTwoFragment(1, 2, 1),
            "occupancy": PowerOfTwoFragment(1, 8, 1),
        }

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        return frozenset(
            {"grid", "warmup", "num_warps", "num_stages", "num_ctas", "occupancy"}
        )


# Mapping from torch dtype to JAX dtype string (e.g., "jnp.float32")
_TORCH_TO_JAX_DTYPE: dict[str, str] = {
    "torch.float16": "jnp.float16",
    "torch.float32": "jnp.float32",
    "torch.float64": "jnp.float64",
    "torch.bfloat16": "jnp.bfloat16",
    "torch.int8": "jnp.int8",
    "torch.int16": "jnp.int16",
    "torch.int32": "jnp.int32",
    "torch.int64": "jnp.int64",
    "torch.uint8": "jnp.uint8",
    "torch.uint32": "jnp.uint32",
    "torch.uint64": "jnp.uint64",
    "torch.bool": "jnp.bool_",
    "torch.complex64": "jnp.complex64",
    "torch.complex128": "jnp.complex128",
    "torch.float8_e4m3fn": "jnp.float8_e4m3fn",
    "torch.float8_e4m3fnuz": "jnp.float8_e4m3fnuz",
    "torch.float8_e5m2": "jnp.float8_e5m2",
    "torch.float8_e5m2fnuz": "jnp.float8_e5m2fnuz",
    "torch.float8_e8m0fnu": "jnp.float8_e8m0fnu",
}


# TPU does not natively support 64-bit element types.
_PALLAS_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


class SliceAddressing(enum.Enum):
    """How a dynamic-offset slice on a tensor dim must be emitted on TPU."""

    DIRECT = enum.auto()  # offset used as-is -> plain pl.ds
    ALIGNED = enum.auto()  # offset rounded to a sublane tile -> aligned-enclosing


def _slice_addressing(
    tensor: torch.Tensor, dim: int, lane_block: int | None = None
) -> SliceAddressing:
    """Whether a dynamic slice on ``dim`` can take any offset.

    TPU only tiles the last two dims into (8, 128) blocks, so a slice on an
    earlier row-major dim reads any offset (DIRECT).  A sublane-dim slice must
    align to a tile boundary (ALIGNED), except f32 over a single lane tile
    (``lane_block`` <= 128) stays contiguous and reads any offset too (DIRECT).
    ``lane_block`` is the last-dim extent (block size, or full width if untiled);
    None stays conservative (ALIGNED).
    """
    if dim < tensor.ndim - 2:
        return SliceAddressing.DIRECT  # major dim: row-major, any offset
    if dim == tensor.ndim - 2:  # 2nd-minor (sublane) dim
        # f32 fills a lane, so a single lane tile is contiguous and reads any
        # offset; bf16 packs two rows per sublane and always needs alignment.
        if (
            tensor.dtype == torch.float32
            and isinstance(lane_block, int)
            and lane_block <= 128
        ):
            return SliceAddressing.DIRECT
        return SliceAddressing.ALIGNED
    return SliceAddressing.ALIGNED  # TODO(tcombes): align lane dim to 128, not sublane


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend for TPU."""

    @property
    def name(self) -> str:
        return "pallas"

    @staticmethod
    # Overrides Backend.map_dot_precision.
    def map_dot_precision(precision: DotPrecision) -> str:
        """Map Helion dot precision to Pallas-specific precision string.

        Pallas/TPU has limited support for different precisions, often
        falling back to the highest available precision.
        """
        pallas_precision_by_dot_precision = {
            "default": "default",
            # "high" is mapped to "highest" because Pallas/Mosaic doesn't yet
            # support it on TPU.
            "high": "highest",
            "highest": "highest",
            "tf32": "highest",
            "tf32x3": "highest",
            "ieee": "highest",
        }
        return pallas_precision_by_dot_precision.get(precision, "default")

    @property
    def max_tensor_numel(self) -> int | None:
        # No compile-time element cap on Pallas; VMEM byte budget is the
        # real constraint and is enforced separately at runtime.
        return None

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        return False

    def max_reduction_threads(self) -> int | None:
        return None

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_JAX_DTYPE:
            raise ValueError(f"Unsupported dtype for Pallas backend: {dtype}")
        return _TORCH_TO_JAX_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        # Promote half-precision types to float32 for numerical stability
        if dtype in (torch.float16, torch.bfloat16):
            return "jnp.float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_pallas_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "jax": "import jax",
            "jnp": "import jax.numpy as jnp",
            "pl": "from jax.experimental import pallas as pl",
            "lax": "import jax.lax as lax",
            "pltpu": "from jax.experimental.pallas import tpu as pltpu",
            "_default_pallas_launcher": "from helion.runtime import default_pallas_launcher as _default_pallas_launcher",
            "_default_pallas_pipeline_launcher": "from helion.runtime import default_pallas_pipeline_launcher as _default_pallas_pipeline_launcher",
            "_default_pallas_fori_launcher": "from helion.runtime import default_pallas_fori_launcher as _default_pallas_fori_launcher",
            "_default_pallas_compact_worklist_launcher": "from helion.runtime import default_pallas_compact_worklist_launcher as _default_pallas_compact_worklist_launcher",
        }

    # Config keys that Pallas actually uses.  Everything else
    # (pid_type, num_warps, num_stages, maxnreg, indexing, etc.)
    # is GPU-specific and should not be tuned.
    _PALLAS_SUPPORTED_KEYS: frozenset[str] = frozenset(
        {
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "pallas_loop_type",
            "pallas_pre_broadcast",
        }
    )

    def supports_config_key(self, key: str) -> bool:
        return key in self._PALLAS_SUPPORTED_KEYS

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"pl.program_id({dim})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"lax.convert_element_type({expr_str}, {dtype_str})"

    @property
    def range_requires_python_int(self) -> bool:
        return True

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = {lid} * {block_size_var} + jnp.arange(0, {block_size_var}, dtype={dtype})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .device_function import pallas_texpr

        return pallas_texpr(expr)

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.pallas import PallasKernelOverrides

        return PallasKernelOverrides()

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            f"lax.convert_element_type({{x}}, {self.dtype_str(target_dtype)})", x=x
        )

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            from .compile_environment import CompileEnvironment

            if tensor_host_args:
                device_expr = f"{tensor_host_args[0]}.device"
            elif CompileEnvironment.current().settings.pallas_interpret:
                device_expr = "'cpu'"
            else:
                device_expr = "'tpu'"
            # Scalars are passed as 1-dim tensors (shape [1]) rather than
            # 0-dim tensors (shape []) because TPU Pallas Mosaic lowering
            # requires rank >= 1 for all block specs.  A 0-dim input causes:
            #   ValueError: The Pallas TPU lowering currently supports only
            #   blocks of rank >= 1.
            # The kernel dereferences the scalar with ``name[0]`` (see
            # ``scalar_arg_preamble``).
            if isinstance(arg, (TensorSizeArg, TensorStrideArg)):
                from .compile_environment import CompileEnvironment

                idx_dtype = CompileEnvironment.current().index_dtype
                return f"torch.tensor([{host_str}], dtype={idx_dtype!r}, device={device_expr})"
            return f"torch.tensor([{host_str}], dtype=torch.float32 if isinstance({host_str}, float) else torch.int32, device={device_expr})"
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        from .ast_extension import statement_from_string
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            # TPU: scalars are wrapped as 1-dim tensors, index with [0]
            return [statement_from_string(f"{arg.name} = {arg.name}[0]")]
        return []

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        return f"jnp.full([{', '.join(shape_dims)}], {value_expr}, {self.dtype_str(dtype)})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"jnp.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"jnp.broadcast_to({expr}, {shape})"

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        if reduction_type in {"sum", "max", "min", "prod"}:
            return f"jnp.{reduction_type}({input_name}, axis={dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

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
    ) -> str:
        fn = "jnp.argmax" if reduction_type == "argmax" else "jnp.argmin"
        return (
            f"lax.convert_element_type("
            f"{fn}({input_name}, axis={dim}), {self.dtype_str(output_dtype)})"
        )

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        if reduction_type == "argmin":
            better = (
                f"(({value}) < ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        else:
            better = (
                f"(({value}) > ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        return [
            f"{acc} = jnp.where({better}, {value}, {acc})",
            f"{acc_index} = jnp.where({better}, {index}, {acc_index})",
        ]

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"jnp.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"jnp.zeros({shape}, dtype={dtype})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return f"jnp.zeros([0], dtype={dtype})"

    def static_rdim_size(self, numel: int) -> int:
        # Pallas block refs use exact tensor dimensions, so RDIM_SIZE must
        # match (no power-of-2 rounding that would exceed the block ref).
        return numel

    def dynamic_rdim_size_expr(self, expr: str) -> str:
        return expr

    def _get_pallas_required_alignment(
        self, dim_from_end: int, tensor_ndim: int, bitwidth: int
    ) -> int:
        """Requirements documented in https://docs.jax.dev/en/latest/pallas/grid_blockspec.html

        Args:
            dim_from_end (int): The dimension being queried for alignment requirements, indexed from the end. i.e. [... ,2, 1, 0]
            tensor_ndim (int): Amount of dimensions for the tensor.
            bitwidth (int): Bitwidth of tensor elements
        """
        # Cap to 32: wider dtypes (e.g. float64, int64) would cause
        # ZeroDivisionError in 32 // bitwidth.  64-bit types are rejected
        # at runtime, so block spec computation uses 32-bit alignment.
        bitwidth = min(bitwidth, 32)
        if dim_from_end == 0:  # Last dimension
            if tensor_ndim <= 1:
                return 128 * (32 // bitwidth)
            return 128
        if dim_from_end == 1:  # Second to last dimension
            return 8
        return 1  # No requirements for other dimensions

    def sublane_tiling(self, dtype: torch.dtype) -> int:
        """Native sublane (2nd-minor) tile for ``dtype``: f32->8, bf16->16, i8->32.

        The jagged carry slices its emit_pipeline VMEM refs at this
        granularity, and such a ref must be accessed as a *whole* native tile:
        a smaller slice (e.g. 8 rows of a bf16 ref, whose tile is 16) is
        rejected by Mosaic ("E2003: unproven memory access alignment"),
        independent of offset.
        """
        bitwidth = min(dtype.itemsize * 8, 32)
        return 8 * (32 // bitwidth)

    fake_tensor_loads: list[tuple[torch.Tensor, list[object]]]

    def process_fake_tensor_load(
        self,
        tensor: torch.Tensor,
        index: list[object],
    ) -> None:
        if not hasattr(self, "fake_tensor_loads"):
            self.fake_tensor_loads = []
        self.fake_tensor_loads.append((tensor, index))

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        """Enforce TPU alignment on block sizes.

        TPU Pallas requires:
        - 1D last dim: multiple of ``128 * (32 // dtype_bits)``
          (128 for f32, 256 for bf16)
        - 2D+ last dim: multiple of 128
        - 2D+ second-to-last dim: multiple of 8

        When the tensor dimension is smaller than the alignment requirement,
        we set the minimum block size to ``next_power_of_2(tensor_dim)``
        instead.  At runtime the block shape is capped to
        ``min(block_size, tensor_dim)`` which equals the full array
        dimension -- always valid per TPU rules.
        """
        from ..autotuner.config_spec import BlockSizeSpec
        from .ast_extension import ExtendedAST
        from .compile_environment import BlockSizeInfo
        from helion._compiler.compile_environment import _to_sympy
        from helion._compiler.host_function import HostFunction
        from helion._compiler.type_info import SequenceType
        from helion._compiler.type_info import TensorType
        from helion._compiler.type_info import TileIndexType

        host_func = HostFunction.current()

        class TensorTiledAccessAnalyzer(ast.NodeVisitor):
            def __init__(self, backend: PallasBackend) -> None:
                super().__init__()
                self.backend = backend
                self.required_alignments: dict[int, int] = {}
                self.update_requirements_from_fake_tensor_loads()

            def visit_Subscript(self, node: ast.Subscript) -> None:
                assert isinstance(node, ExtendedAST)
                assert isinstance(node.value, ExtendedAST)
                value_type = node.value._type_info
                if not isinstance(value_type, TensorType):
                    return
                tensor = value_type.fake_value
                if isinstance(node.slice, (ast.Tuple, ast.List)):
                    num_squeezed_dimensions = 0
                    for i, subscript in enumerate(node.slice.elts):
                        if (
                            isinstance(subscript, ast.Constant)
                            and subscript.value is None
                        ):
                            num_squeezed_dimensions += 1
                            continue
                        accessed_dim = i - num_squeezed_dimensions
                        self.maybe_update_alignment_requirement(
                            tensor, accessed_dim, subscript
                        )
                else:
                    self.maybe_update_alignment_requirement(tensor, 0, node.slice)
                # Nested subscripts (e.g. idx[tile] in table[idx[tile], :])
                # are themselves tiled accesses and need their own alignment.
                self.generic_visit(node)

            def maybe_update_alignment_requirement(
                self, tensor: torch.Tensor, accessed_dim_start: int, subscript: ast.AST
            ) -> None:
                if not isinstance(subscript, ExtendedAST):
                    return
                subscript_type = subscript._type_info
                tile_index_types: list[TileIndexType] = []
                if isinstance(subscript_type, TileIndexType):
                    tile_index_types.append(subscript_type)
                elif isinstance(subscript_type, SequenceType):
                    for el_type in subscript_type.element_types:
                        if isinstance(el_type, TileIndexType):
                            tile_index_types.append(el_type)

                for i, tile_index_type in enumerate(tile_index_types):
                    bid = tile_index_type.block_id
                    accessed_dim = accessed_dim_start + i
                    dim_from_end = tensor.ndim - accessed_dim - 1
                    bitwidth = tensor.dtype.itemsize * 8

                    required_alignment = self.backend._get_pallas_required_alignment(
                        dim_from_end, tensor.ndim, bitwidth
                    )
                    self.maybe_update_required_alignment(bid, required_alignment)

            def maybe_update_required_alignment(
                self, bid: int, required_alignment: int
            ) -> None:
                if bid not in self.required_alignments:
                    self.required_alignments[bid] = required_alignment
                else:
                    self.required_alignments[bid] = max(
                        self.required_alignments[bid], required_alignment
                    )

            def update_requirements_from_fake_tensor_loads(self) -> None:
                # When tensors are indexed within external lambdas called by the kernel,
                # they generate fake loads, which we don't pickup during AST walk.
                if not hasattr(self.backend, "fake_tensor_loads"):
                    return
                if block_sizes is None:
                    return
                for info in block_sizes:
                    if not isinstance(info, BlockSizeInfo):
                        continue
                    for tensor, subscripts in self.backend.fake_tensor_loads:
                        for dim, subscript in enumerate(subscripts):
                            if isinstance(subscript, torch.SymInt) and info.dim_matches(
                                _to_sympy(subscript)
                            ):
                                dim_from_end = tensor.ndim - 1 - dim
                                bitwidth = tensor.dtype.itemsize * 8
                                required_alignment = (
                                    self.backend._get_pallas_required_alignment(
                                        dim_from_end, tensor.ndim, bitwidth
                                    )
                                )
                                self.maybe_update_required_alignment(
                                    info.block_id, required_alignment
                                )

        analyzer = TensorTiledAccessAnalyzer(self)
        for stmt in host_func.body:
            analyzer.visit(stmt)

        from torch._inductor.runtime.runtime_utils import next_power_of_2

        if block_sizes is not None and kernel_tensor_sizes is not None:
            for shape in kernel_tensor_sizes:
                for bid, info in enumerate(block_sizes):
                    if not isinstance(info, BlockSizeInfo):
                        continue
                    # pyrefly: ignore[no-matching-overload]
                    if math.prod(shape) == info.var:
                        # avoid creating size-1 kernel tensors, which triggers Pallas Mosaic lowering failure:
                        # https://github.com/jax-ml/jax/issues/36970
                        analyzer.maybe_update_required_alignment(bid, 2)

        for spec in block_specs:
            if not isinstance(spec, BlockSizeSpec):
                continue
            bid = spec.block_ids[0]
            if bid not in analyzer.required_alignments:
                continue
            requirement_alignment = analyzer.required_alignments[bid]
            dim_size = next_power_of_2(max(spec.size_hint, 1))
            # Cap the alignment requirement by the tensor lane dim: when
            # the dim is smaller than the requirement, the full-dim access
            # is always aligned at offset 0 so block_size = dim_size is
            # safe.  When the dim is at least as big as the requirement,
            # ``min`` returns ``requirement_alignment`` and the strict
            # floor still applies (used by aot_example.sum_aot, n=256).
            spec.update_min(min(requirement_alignment, dim_size))

        # Propagate alignment minimums from inner tiles to their bounding outer tiles.
        block_specs_by_id = {
            spec.block_ids[0]: spec
            for spec in block_specs
            if isinstance(spec, BlockSizeSpec)
        }
        for spec in block_specs_by_id.values():
            bounded_by = spec.bounded_by_block_id
            if bounded_by is None:
                continue
            outer_spec = block_specs_by_id.get(bounded_by)
            if outer_spec is not None:
                outer_spec.update_min(spec.min_size)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        return {}

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        from ..autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        from ..autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def get_paired_device_micros_bench(
        self,
    ) -> Callable[..., list[tuple[float, float]]] | None:
        """Pallas ``jax.profiler`` device-µs bench for the final-pick re-rank.

        Returns None (keeping the wall-clock rebench) when the user opts out via
        ``HELION_AUTOTUNE_PALLAS_RANK_BY=wall_time`` or ``jax`` is unavailable.
        """
        from ..autotuner.benchmarking import make_pallas_paired_device_micros_bench

        return make_pallas_paired_device_micros_bench()

    def supports_precompile(self) -> bool:
        return False

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # Pallas/JAX compilation and runtime errors are generally expected
        # during autotuning when invalid configs are tried.
        # Only truly fatal errors (KeyboardInterrupt, SystemExit, etc.)
        # should propagate; everything else is a config incompatibility.
        if isinstance(err, Exception):
            return "debug"
        return None

    def rng_seed_buffer_expr(self, count: int) -> str:
        # Generate on CPU, then move to the accelerator so the full 64-bit
        # Philox seed survives backend handoff.
        return f"inductor_prims.seeds({count}, torch.device('cpu')).to(torch.accelerator.current_accelerator())"

    def _compute_block_spec_info(
        self,
        sorted_args: list[Argument] | None,
        config: Config,
    ) -> (
        list[
            tuple[
                tuple[int | None, ...],
                tuple[int | tuple[int, int, int] | None, ...],
            ]
            | None
        ]
        | None
    ):
        """Compute per-tensor ``(block_shape, grid_dims)`` from codegen tiling info.

        Uses ``DeviceFunction.pallas_tensor_dim_tilings`` (recorded during
        ``plan_tiling`` from SymInt subscripts) for an unambiguous
        dim → block_id mapping.
        """
        if sorted_args is None:
            return None

        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction
        from .device_function import SymbolArgument
        from .device_function import TensorArg
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg
        from .host_function import HostFunction
        from .program_id import FlatProgramIDs

        env = CompileEnvironment.current()
        device_fn = DeviceFunction.current()

        # Build block_id → grid_dim from the actual PID ordering (which
        # reflects loop_order).  ``pid_info`` is ordered by grid dimension,
        # so pid_info[g].block_id is the block_id assigned to grid dim g.
        if device_fn.pid is None:
            return None
        flat_grid_block_ids = [pid.block_id for pid in device_fn.pid.pid_info]
        block_id_to_grid_dim = {bid: g for g, bid in enumerate(flat_grid_block_ids)}
        known_block_ids = set(block_id_to_grid_dim)

        # FlattenedTileStrategy collapses all block_ids into a single
        # pid_info entry, but the full set lives in device_ir.grid_block_ids.
        # Recover them so we can build flat decomposition and so downstream
        # checks (e.g. 1D tensor validation) see every block_id.
        flat_decomp: dict[int, tuple[int, int, int]] | None = None
        if isinstance(device_fn.pid, FlatProgramIDs):
            device_ir = HostFunction.current().device_ir
            all_grid_block_ids = [
                bid for bids in device_ir.grid_block_ids for bid in bids
            ]
            known_block_ids.update(all_grid_block_ids)

            if len(all_grid_block_ids) > 1:
                import sympy

                stride = 1
                flat_decomp = {}
                for bid in all_grid_block_ids:
                    bs = env.block_sizes[bid].from_config(config)
                    numel = env.block_sizes[bid].numel
                    if not isinstance(bs, int) or isinstance(numel, str):
                        return None
                    try:
                        numel_val = (
                            int(numel) if isinstance(numel, sympy.Expr) else numel
                        )
                    except (TypeError, ValueError):
                        return None
                    num_blocks = -(-numel_val // bs)  # cdiv
                    flat_decomp[bid] = (0, stride, num_blocks)
                    stride *= num_blocks

        result: list[
            tuple[tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]]
            | None
        ] = []

        for arg in sorted_args:
            if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
                result.append(None)  # scalars wrapped as 1-D tensors
                continue
            if not isinstance(arg, TensorArg):
                continue
            if arg.fake_value.ndim == 0:
                result.append(None)
                continue
            tensor = arg.fake_value
            dim_tilings = device_fn.pallas_tensor_dim_tilings.get(id(tensor))
            if dim_tilings is None:
                # this means this tensor isn't accessed at all in the kernel
                result.append(None)
                return None
            block_shape: list[int | None] = []
            grid_dims: list[int | tuple[int, int, int] | None] = []
            for d in range(tensor.ndim):
                dim_tiling = dim_tilings[d]
                if not dim_tiling.can_tile or len(dim_tiling.block_ids) == 0:
                    block_shape.append(None)
                    grid_dims.append(None)
                    continue
                assert len(dim_tiling.block_ids) == 1
                bid = dim_tiling.block_ids[0]
                if bid is not None and bid in known_block_ids:
                    bs = env.block_sizes[bid].from_config(config)
                    if isinstance(bs, int):
                        block_shape.append(bs)
                        dim_size = tensor.shape[d]
                        # When the block covers the entire tensor
                        # dimension there is only one tile, so the grid
                        # index must be constant 0 — iterating would
                        # read out-of-bounds (e.g. bias [1, N] with
                        # block_size > 1).
                        if isinstance(dim_size, int) and dim_size <= bs:
                            grid_dims.append(None)
                        elif flat_decomp is not None and bid in flat_decomp:
                            grid_dims.append(flat_decomp[bid])
                        else:
                            grid_dims.append(block_id_to_grid_dim[bid])
                        continue
                block_shape.append(None)
                grid_dims.append(None)
            result.append((tuple(block_shape), tuple(grid_dims)))
        return result

    def _compute_pad_info(
        self,
        sorted_args: list[Argument] | None,
        config: Config,
    ) -> list[tuple[int, int, int, int]] | None:
        """Identify pl.ds() dims that may need padding and their block sizes.

        Uses ``pallas_pad_info`` recorded during codegen to identify which
        tensor dimensions use ``pl.ds()`` slicing.

        Returns ``[(arg_index, tensor_dim, block_size, extra_pad), ...]``
        or ``None``.  The launcher computes the actual pad amount at runtime
        as ``(-tensor.shape[dim]) % block_size + extra_pad``.

        ``extra_pad`` is 0 when the tile loop starts at offset 0,
        ``begin % block_size`` for a constant begin offset, or
        ``block_size - 1`` for a data-dependent begin.
        """
        if sorted_args is None:
            return None

        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction
        from .device_function import TensorArg

        env = CompileEnvironment.current()
        device_fn = DeviceFunction.current()
        if not device_fn.pallas_pad_info:
            return None

        result: list[tuple[int, int, int, int]] = []
        for i, arg in enumerate(sorted_args):
            if not isinstance(arg, TensorArg):
                continue
            dims_info = device_fn.pallas_pad_info.get(id(arg.fake_value))
            if dims_info is not None:
                for dim, (block_id, extra_pad) in dims_info.items():
                    bsi = env.block_sizes[block_id]
                    bs = bsi.from_config(config)
                    if isinstance(bs, int) and bs > 1:
                        result.append((i, dim, bs, extra_pad))

        return result or None

    def _detect_matmul_dot_general_lowering(
        self,
        *,
        sorted_args: list[Argument] | None,
        config: Config,
        output_indices: list[int],
        inplace_indices: list[int],
        block_spec_info: object,
    ) -> dict[str, object] | None:
        """Detect a pure-matmul, no-tiling kernel the launcher can lower as
        ``jax.jit(lax.dot_general(...))`` instead of ``pl.pallas_call(...)``.

        Eligible when: 2 input tensors + 1 output-only tensor; all 2D with
        matching M/K/N contiguous layout (BMM not covered yet); the device IR
        has one ``aten.mm``/``addmm`` family op; and the picked block sizes
        cover every dim (single launch, no inner K tile).  Returns the spec
        dict consumed by ``_build_matmul_dot_general_jit_fn``, else ``None``.
        """
        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction
        from .device_function import TensorArg
        from .host_function import HostFunction

        if sorted_args is None or not output_indices:
            return None
        # Pure-output kernels only (no in-place mutation, single output).
        if inplace_indices or len(output_indices) != 1:
            return None

        # Exactly 2 inputs + 1 output, all tensors (a scalar arg means it isn't
        # a pure ``out = matmul(x, y)``).
        tensor_positions = [
            i for i, arg in enumerate(sorted_args) if isinstance(arg, TensorArg)
        ]
        if len(sorted_args) != 3 or len(tensor_positions) != 3:
            return None

        out_pos = output_indices[0]
        input_positions = [p for p in tensor_positions if p != out_pos]
        if len(input_positions) != 2:
            return None

        lhs_arg = sorted_args[input_positions[0]]
        rhs_arg = sorted_args[input_positions[1]]
        out_arg = sorted_args[out_pos]
        assert isinstance(lhs_arg, TensorArg)
        assert isinstance(rhs_arg, TensorArg)
        assert isinstance(out_arg, TensorArg)
        lhs_t = lhs_arg.fake_value
        rhs_t = rhs_arg.fake_value
        out_t = out_arg.fake_value
        # 2D matmul, matching contraction dim, statically-known shapes.
        if lhs_t.ndim != 2 or rhs_t.ndim != 2 or out_t.ndim != 2:
            return None
        try:
            m = int(lhs_t.shape[0])
            k_lhs = int(lhs_t.shape[1])
            k_rhs = int(rhs_t.shape[0])
            n = int(rhs_t.shape[1])
            out_m = int(out_t.shape[0])
            out_n = int(out_t.shape[1])
        except (TypeError, ValueError):
            return None
        if k_lhs != k_rhs or out_m != m or out_n != n:
            return None

        # The device IR must contain an aten.mm/addmm/bmm family op
        # (via the shared ``_loop_contains_matmul`` predicate).
        device_fn = DeviceFunction.current()
        device_ir = HostFunction.current().device_ir
        if not device_ir.grid_block_ids:
            return None
        # Any root-grid loop containing a matmul qualifies.
        matmul_present = any(
            _loop_contains_matmul(device_fn, list(grid_block_ids))
            for grid_block_ids in device_ir.grid_block_ids
        )
        if not matmul_present:
            return None

        # Orient to lhs=(M, K), rhs=(K, N); the user may have written
        # ``f(y, x) -> x @ y``. For all-equal dims either ordering is the same.
        if lhs_t.shape == (m, k_lhs) and rhs_t.shape == (k_lhs, n):
            lhs_arg_pos, rhs_arg_pos = input_positions
            lhs_resolved, rhs_resolved = lhs_t, rhs_t
        elif lhs_t.shape == (k_lhs, n) and rhs_t.shape == (m, k_lhs):
            rhs_arg_pos, lhs_arg_pos = input_positions
            lhs_resolved, rhs_resolved = rhs_t, lhs_t
        else:
            return None

        # Every block size must be >= max(M, N, K): a smaller block means a
        # multi-launch (tiled) kernel, not the no-tiling case.
        env = CompileEnvironment.current()
        max_dim = max(m, k_lhs, n)
        for bsi in env.block_sizes:
            if bsi is None:  # type: ignore[unreachable]
                continue
            try:
                bs = bsi.from_config(config)
            except Exception:
                return None
            if not isinstance(bs, int) or bs < max_dim:
                return None

        # Every tensor must be fully untiled (all grid_dims None); outer-grid
        # BlockSpecs still need pl.pallas_call.
        if block_spec_info is None or not isinstance(block_spec_info, list):
            return None
        for pos in (input_positions[0], input_positions[1], out_pos):
            if pos >= len(block_spec_info):
                return None
            entry = block_spec_info[pos]
            if entry is None:
                return None
            block_shape, grid_dims = entry
            if any(gd is not None for gd in grid_dims):
                return None

        # All checks passed; build the launcher spec. bf16/fp16 output from an
        # f32 accumulator needs preferred f32 + cast-back; f32 is already f32.
        f32_acc = out_t.dtype in (torch.bfloat16, torch.float16)
        # Map positions to the launcher's tensor-arg order (sorted non-output
        # positions; see ``_pallas_prepare_args``).
        non_output_positions = sorted(p for p in tensor_positions if p != out_pos)
        return {
            "lhs_tensor_arg_index": non_output_positions.index(lhs_arg_pos),
            "rhs_tensor_arg_index": non_output_positions.index(rhs_arg_pos),
            "lhs_dtype": self.dtype_str(lhs_resolved.dtype),
            "rhs_dtype": self.dtype_str(rhs_resolved.dtype),
            "out_dtype": self.dtype_str(out_t.dtype),
            "f32_accumulator": bool(f32_acc),
        }

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
        # Determine which arg positions are outputs.  A tensor is an output if:
        #   1. It was created inside the function body (not in input_sources), OR
        #   2. It is a function parameter that is mutated in-place (e.g. x[tile] += ...)
        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction
        from .device_function import TensorArg
        from .host_function import HostFunction

        device_fn = DeviceFunction.current()

        def _empty_allocated_vars(body: list[ast.stmt]) -> set[str]:
            """Return names of variables allocated with torch.empty/empty_like/new_empty.

            Only checks top-level assignments; allocations nested inside
            if/with/try are conservatively missed (treated as needing input,
            which is correct but suboptimal).
            """
            result: set[str] = set()
            for stmt in body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr in ("empty", "empty_like", "new_empty")
                ):
                    result.add(stmt.targets[0].id)
            return result

        output_indices: list[int] = []
        # Indices of output tensors that are also read by the kernel
        # (inplace-mutated params or body-created tensors the kernel reads).
        # These must use VMEM BlockSpecs. Output-only tensors (written but
        # never read) get HBM in_specs to avoid VMEM pressure.
        inplace_indices: list[int] = []
        if sorted_args is not None:
            env = CompileEnvironment.current()
            host_fn = HostFunction.current()
            read_names, write_names = device_fn.get_tensor_read_write_names()
            mutated_params = write_names & {a.arg for a in host_fn.args.args}
            input_storages = {id(t.untyped_storage()) for t in env.input_sources}
            # Only tensors allocated with torch.empty/empty_like/new_empty can be
            # output-only — their initial values are undefined, so it's safe
            # to use HBM BlockSpecs.  Tensors allocated with torch.zeros_like,
            # torch.full, etc. have meaningful initial values that must be
            # preserved via VMEM BlockSpecs.
            empty_vars = _empty_allocated_vars(host_fn.body)
            for i, arg in enumerate(sorted_args):
                if not isinstance(arg, TensorArg):
                    continue
                arg_name = arg.host_str()
                if (
                    id(arg.fake_value.untyped_storage()) not in input_storages
                    and arg_name in write_names
                ):
                    # Tensor created inside the function body (output)
                    output_indices.append(i)
                    if arg_name in read_names or arg_name not in empty_vars:
                        # Also read by the kernel (e.g. broadcast result)
                        inplace_indices.append(i)
                elif arg_name in mutated_params:
                    # Input tensor mutated in-place
                    output_indices.append(i)
                    inplace_indices.append(i)

        # Collect output-only tensor names so codegen can retarget their
        # allocations to ``device='meta'`` and capture the launcher return.
        output_only_set = set(output_indices) - set(inplace_indices)
        output_only_names: list[str] = []
        if sorted_args is not None:
            for i in output_indices:
                if i in output_only_set:
                    arg = sorted_args[i]
                    assert isinstance(arg, TensorArg)
                    output_only_names.append(arg.host_str())
        self._output_only_names = output_only_names

        launcher_args = [*args]
        if has_rng_ops:
            launcher_args.append("_rng_seed_buffer")
        launcher_args.extend(
            [f"_output_indices={output_indices}", f"_inplace_indices={inplace_indices}"]
        )

        block_spec_info = self._compute_block_spec_info(sorted_args, config)
        if block_spec_info is not None:
            if has_rng_ops:
                block_spec_info.append(None)  # RNG seed buffer is untiled
            launcher_args.append(f"_block_spec_info={block_spec_info!r}")

        pad_info = self._compute_pad_info(sorted_args, config)
        if pad_info:
            launcher_args.append(f"_ds_pad_dims={pad_info!r}")

        from .device_function import PallasMemorySpace

        mem_space = device_fn.pallas_memory_space
        if sorted_args is not None:
            smem_arg_indices = [
                i
                for i, arg in enumerate(sorted_args)
                if isinstance(arg, TensorArg)
                and mem_space.get(id(arg.fake_value)) == PallasMemorySpace.SMEM
            ]
            if smem_arg_indices:
                launcher_args.append(f"_smem_arg_indices={smem_arg_indices!r}")

        # Pass scratch shapes for pipeline/fori_loop launcher
        pallas_loop_type = config.get("pallas_loop_type", "unroll")
        if pallas_loop_type in ("emit_pipeline", "fori_loop", "compact_worklist"):
            scratch_shapes = [
                (
                    s.shape,
                    self.dtype_str(s.dtype) if s.dtype is not None else None,
                    s.scratch_type,
                )
                for s in device_fn._scratch_args
            ]
            if scratch_shapes:
                launcher_args.append(f"_scratch_shapes={scratch_shapes!r}")

            # Identify which launcher arg positions correspond to pipeline-body
            # tensors (need HBM refs); all others get proper BlockSpecs.
            from .device_function import TensorArg

            if sorted_args is not None:
                pipeline_arg_indices = [
                    i
                    for i, arg in enumerate(sorted_args)
                    if isinstance(arg, TensorArg)
                    and mem_space.get(id(arg.fake_value)) == PallasMemorySpace.HBM
                ]
                if pipeline_arg_indices:
                    launcher_args.append(
                        f"_pipeline_arg_indices={pipeline_arg_indices!r}"
                    )

        if CompileEnvironment.current().settings.pallas_interpret:
            launcher_args.append("_pallas_interpret=True")

        # No-tiling pure 2D matmul: emit ``_matmul_dot_general=...`` so the
        # launcher uses ``jax.jit(lax.dot_general(...))`` instead of
        # ``pl.pallas_call(...)``. XLA can then attach cross_program_prefetch,
        # closing the ~12% gap to ``jnp.matmul`` that ``tpu_custom_call``
        # opacity imposes. Falls back silently when ineligible.
        matmul_spec = self._detect_matmul_dot_general_lowering(
            sorted_args=sorted_args,
            config=config,
            output_indices=output_indices,
            inplace_indices=inplace_indices,
            block_spec_info=block_spec_info,
        )
        if matmul_spec is not None:
            launcher_args.append(f"_matmul_dot_general={matmul_spec!r}")

        if pallas_loop_type == "compact_worklist" and sorted_args is not None:
            launcher_args.extend(self._compact_worklist_launcher_args(sorted_args))

        return launcher_args

    def _compact_worklist_launcher_args(self, sorted_args: list[Argument]) -> list[str]:
        """Emit the compact-worklist-specific launcher kwargs.

        ``_build_worklist`` is the module-level jnp builder (emitted in
        generate_ast); the offset arg indices map its params to host-call arg
        positions; the metadata fields + owner-ref position drive scalar-prefetch
        selection and the owner-indexed BlockSpec index_maps.
        """
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg
        from .pallas.compact_worklist import metadata_field_names

        env = CompileEnvironment.current()
        plan = env.compact_worklist_plan
        assert plan is not None

        name_to_index: dict[str, int] = {}
        for i, arg in enumerate(sorted_args):
            if isinstance(arg, TensorArg):
                name_to_index[arg.host_str()] = i
        offset_indices = [name_to_index[n] for n in env.compact_worklist_offset_params]
        fields = metadata_field_names(plan)
        # Compact-tile tensors (aligned load + exact store) both get a per-tile
        # pl.Element BlockSpec sliced at tile_start, so Pallas double-buffers BOTH
        # the load prefetch and the store write-back across work items.
        #
        # The store is a masked full-block write.  The two robust EXACT-store
        # alternatives were both worse/unavailable here: (a) staging VMEM +
        # make_async_copy over pl.ds(tile_start, tile_extent) serializes (~1.8x
        # slower: 4.5ms vs 2.5ms) because a straight-line compact tile has no inner
        # loop to overlap; (b) a pl.BoundedSlice store BlockSpec (exact + double-buffered)
        # is rejected by this JAX's Mosaic lowering ("Unsupported block dimension
        # type: BoundedSlice" -- it only works inside pltpu.emit_pipeline).  The
        # full-block write's only hazard is a partial last tile overlapping the
        # next sequence's leading rows; "arbitrary" dimension semantics serialize
        # that grid-ordered overlap so the later, correct write wins (verified
        # bitwise == fori_loop across uniform/partial/unaligned/jagged + 5 random
        # seeds).  Robust+fast exact store == the deferred emit_pipeline +
        # pl.BoundedSlice path.
        aligned_indices = [
            name_to_index[p.arg_name]
            for p in plan.tensor_policies
            if p.kind in ("compact_aligned_load", "compact_exact_store")
            and p.arg_name in name_to_index
        ]
        return [
            "_compact_build_worklist=_build_worklist",
            f"_compact_offset_arg_indices={offset_indices!r}",
            f"_compact_metadata_fields={fields!r}",
            "_compact_owner_ref_pos=0",
            f"_compact_num_scalar_prefetch={len(fields)}",
            f"_compact_aligned_arg_indices={aligned_indices!r}",
            f"_compact_tile_start_ref_pos={fields.index('tile_starts')}",
            f"_compact_block={env.compact_worklist_block}",
        ]

    def build_launcher_name(self, config: Config) -> str:
        """Return the launcher name to use based on ``pallas_loop_type``."""
        from ..autotuner.config_spec import VALID_PALLAS_LOOP_TYPES

        pallas_loop_type = config.get("pallas_loop_type", "unroll")
        if pallas_loop_type not in VALID_PALLAS_LOOP_TYPES:
            raise ValueError(
                f"Invalid pallas_loop_type {pallas_loop_type!r}. "
                f"Expected one of {VALID_PALLAS_LOOP_TYPES}."
            )
        if pallas_loop_type == "emit_pipeline":
            return "_default_pallas_pipeline_launcher"
        if pallas_loop_type == "fori_loop":
            return "_default_pallas_fori_launcher"
        if pallas_loop_type == "compact_worklist":
            # Detection + plan stash happen in pre_codegen; route to the compact
            # launcher (builds the worklist in-jit -> dynamic num_work grid with
            # scalar-prefetch metadata).
            return "_default_pallas_compact_worklist_launcher"
        return self.default_launcher_name

    def get_launcher_name(self) -> str:
        """Return the launcher name based on the current config."""
        from .device_function import DeviceFunction
        from .device_function import NoCurrentFunction

        try:
            device_fn = DeviceFunction.current()
        except NoCurrentFunction:
            return self.default_launcher_name
        return self.build_launcher_name(device_fn.config)

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .compile_environment import CompileEnvironment
        from .pallas.plan_tiling import plan_tiling

        plan_tiling(graphs, config, tile_strategy)

        # compact_worklist_* is per-CONFIG state, but one CompileEnvironment is
        # reused across all configs of a BoundKernel (see CompileEnvironment's
        # "no config-specific state" contract).  Reset before re-detecting so a
        # later non-compact config never inherits a prior compact config's plan
        # -- many lowering paths gate on ``env.compact_worklist_plan is not None``
        # (PID strategy, loop-bound remap, fori handling, ds slicing), not on
        # pallas_loop_type, so a stale plan would mis-lower a fori/emit config.
        env = CompileEnvironment.current()
        env.compact_worklist_plan = None
        env.compact_worklist_upper = 1
        env.compact_worklist_block = 1
        env.compact_worklist_offset_params = []

        if config.get("pallas_loop_type") == "compact_worklist":
            self._setup_compact_worklist(config)

    def _setup_compact_worklist(self, config: Config) -> None:
        """Detect + stash the compact-worklist plan before device codegen.

        Runs early (pre_codegen) so ``env.compact_worklist_plan`` is set when the
        grid strategy selects ``WorklistProgramIDs`` and the inner loop remaps its
        begin/end to metadata refs.  Registers the N metadata ref names as
        ``wrapper_only_params`` (kernel-signature-only) and computes the static
        megablocks ``UPPER``.  ``detect_*`` raises ``exc.InvalidConfig`` on a
        non-matching kernel (autotuner-skippable).
        """
        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction
        from .host_function import HostFunction
        from .pallas.compact_worklist import detect_compact_worklist_plan
        from .pallas.compact_worklist import metadata_arg_names

        env = CompileEnvironment.current()
        host_fn = HostFunction.current()
        plan = detect_compact_worklist_plan(host_fn)
        env.compact_worklist_plan = plan

        device_fn = DeviceFunction.current()
        for name in metadata_arg_names(plan):
            ref = f"{name}_ref"
            if ref not in device_fn.wrapper_only_params:
                device_fn.wrapper_only_params.append(ref)

        # Compact-axis tile block size (NOT max(block_sizes): for fully jagged
        # with Q_BLOCK < KV_BLOCK that would undersize the worklist metadata).
        compact_block = env.block_sizes[plan.compact_axis.block_id].from_config(config)
        assert compact_block is not None, "compact tile has no block size"
        env.compact_worklist_block = int(compact_block)
        env.compact_worklist_upper = self._compact_worklist_upper(plan, config, host_fn)

    def _compact_worklist_upper(
        self, plan: CompactWorklistPlan, config: Config, host_fn: HostFunction
    ) -> int:
        """Static UPPER: the padded length of the worklist metadata arrays.

        Must be >= the worst-case ``num_work = sum_owners cdiv(length, BLOCK)``,
        else the dynamic Pallas grid indexes past the scalar-prefetch metadata
        (``jnp.repeat(total_repeat_length=UPPER)`` would silently truncate the
        worklist).  Detection only accepts the packed-offsets idiom (store
        safety), so owner ranges are contiguous/non-overlapping
        (``sum(length) == total``) and the tight megablocks bound
        ``cdiv(total, BLOCK) + num_owners - 1`` provably holds.  All terms are
        concrete ints under ``static_shapes=True``.
        """
        from ..runtime.compact_worklist import packed_upper_bound
        from .compile_environment import CompileEnvironment

        params = dict(host_fn.params.arguments)
        # Owner count from the captured grid bound (e.g. q_offsets.shape[0] - 1).
        # num_owners_expr is a codegen-derived host expression; if it references a
        # name not in params (a source shape we failed to inline), surface it as
        # an autotuner-skippable InvalidConfig rather than a bare exception that
        # would abort the whole search.
        try:
            num_owners = int(eval(plan.num_owners_expr, {}, params))
        except Exception as e:
            raise exc.InvalidConfig(
                f"compact_worklist: could not evaluate owner-count expression "
                f"{plan.num_owners_expr!r}: {e}"
            ) from e
        # total_compact = leading dim of the compact_aligned_load tensor.
        compact_arg = next(
            p.arg_name for p in plan.tensor_policies if p.kind == "compact_aligned_load"
        )
        total = int(params[compact_arg].shape[0])
        block = CompileEnvironment.current().compact_worklist_block
        # Single source of the tight megablocks bound (also unit-tested).
        return packed_upper_bound(total, num_owners, block)


def _detect_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
) -> bool:
    """Check if a device loop contains a matmul with MMA-compatible dtypes.

    Returns True only when the loop contains a compatible addmm/dot AND
    the grid has at least 2 block IDs (M and N), so the MMA pipeline
    can map them to tile offsets.  Three-level loops (grid[M] +
    device_loop[N] + device_loop[K]) are NOT supported yet.
    """
    from ..language._decorators import is_api_func
    from .cute.cute_mma import can_codegen_cute_mma_aten
    from .cute.cute_mma import can_codegen_cute_mma_dot
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction

    # MMA lowering currently relies on a single grid state that carries
    # both the M and N axes. Nested grid loops like grid[M] + grid[N] do
    # not satisfy that requirement because GenerateAST.current_grid_state
    # only tracks the innermost grid.
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    if len(device_ir.grid_block_ids[0]) != 2:
        return False
    root_grid_ids = set(device_ir.grid_block_ids[0])
    # CuTe MMA fragment partitioning is currently keyed to physical threads.
    # When an M/N tile is partially serialized into lane loops, the same
    # fragment would be reused for multiple logical lanes and produce
    # incorrect results. A pure K reduction loop is different: it does not
    # contribute MMA fragment coordinates, so we can still enable mma_mode
    # there to suppress synthetic lane loops around the K body.
    if any(
        block_id in root_grid_ids and threads > 0 and threads < block_size
        for block_id, block_size, threads in zip(
            block_ids,
            block_sizes,
            num_threads_config,
            strict=False,
        )
    ):
        return False
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        if graph_info.block_ids != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            # Only addmm/baddbmm trigger MMA mode — mm/bmm don't have
            # a built-in accumulator so their result is needed per iteration.
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                return True
    return False


def _largest_divisor_at_most(size: int, limit: int) -> int:
    for divisor in range(limit, 0, -1):
        if size % divisor == 0:
            return divisor
    return 1


def _detect_specialized_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    config: Config,
) -> bool:
    from ..language._decorators import is_api_func
    from .compile_environment import CompileEnvironment
    from .cute.cute_mma import _choose_mma_impl
    from .cute.cute_mma import _mma_active_n_threads
    from .cute.cute_mma import _tcgen05_root_m_threads
    from .cute.cute_mma import can_codegen_cute_mma_aten
    from .cute.cute_mma import can_codegen_cute_mma_dot
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 2:
        return False
    if len(block_ids) != 1 or any(block_id in root_grid_ids for block_id in block_ids):
        return False

    env = CompileEnvironment.current()
    root_block_sizes: list[int] = []
    root_thread_counts: list[int] = []
    root_thread_auto: list[bool] = []
    for block_id in root_grid_ids:
        block_size = env.block_sizes[block_id].from_config(config)
        if not isinstance(block_size, int):
            return False
        root_block_sizes.append(block_size)
        threads = env.config_spec.num_threads.config_get(
            config.num_threads, block_id, 0
        )
        resolved_threads = threads if threads > 0 else block_size
        root_thread_counts.append(resolved_threads)
        root_thread_auto.append(threads == 0)

    if functools.reduce(operator.mul, root_thread_counts, 1) > 1024:
        for idx in sorted(
            (i for i, is_auto in enumerate(root_thread_auto) if is_auto),
            reverse=True,
        ):
            other_threads = functools.reduce(
                operator.mul,
                (
                    root_thread_counts[j]
                    for j in range(len(root_thread_counts))
                    if j != idx
                ),
                1,
            )
            if other_threads <= 0:
                continue
            thread_budget = max(1024 // other_threads, 1)
            next_threads = _largest_divisor_at_most(
                root_block_sizes[idx], thread_budget
            )
            root_thread_counts[idx] = next_threads
            if functools.reduce(operator.mul, root_thread_counts, 1) <= 1024:
                break

    (bk,) = block_sizes
    if not isinstance(bk, int):
        return False
    bm, bn = root_block_sizes
    root_m_threads, root_n_threads = root_thread_counts

    def root_threads_support_impl(mma_impl: str) -> bool:
        if mma_impl == "tcgen05":
            mma_n_threads = _mma_active_n_threads("tcgen05")
            min_root_m_threads = _tcgen05_root_m_threads(bm, bn)
            if (
                root_m_threads < min_root_m_threads
                or root_m_threads > bm
                or bm % root_m_threads != 0
            ):
                return False
            if root_n_threads < mma_n_threads or root_n_threads > bn:
                return False
            if bn % root_n_threads != 0:
                return False
            return root_m_threads * root_n_threads <= 1024
        if mma_impl == "warp":
            return root_m_threads == bm and root_n_threads == bn
        return False

    for graph_info in fn.codegen.codegen_graphs:
        if getattr(graph_info, "block_ids", None) != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                lhs_node = node.args[1]
                if not isinstance(lhs_node, torch.fx.Node):
                    continue
                lhs_val = lhs_node.meta.get("val")
                if not isinstance(lhs_val, torch.Tensor):
                    continue
                mma_impl = _choose_mma_impl(
                    lhs_val.dtype, bm=bm, bn=bn, bk=bk, config=config
                )
                if mma_impl != "universal" and root_threads_support_impl(mma_impl):
                    return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                lhs_node = node.args[0]
                if not isinstance(lhs_node, torch.fx.Node):
                    continue
                lhs_val = lhs_node.meta.get("val")
                if not isinstance(lhs_val, torch.Tensor):
                    continue
                mma_impl = _choose_mma_impl(
                    lhs_val.dtype, bm=bm, bn=bn, bk=bk, config=config
                )
                if mma_impl != "universal" and root_threads_support_impl(mma_impl):
                    return True
    return False


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


def _attention_loop_shape(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> tuple[int, int, AttentionScorePlan] | None:
    """Recognize the flash-attention dataflow in a tile_n device loop.

    Matches BY SHAPE (not exclusivity), mirroring the spike sequence: a
    ``ForLoopGraphInfo`` over the key tile whose body holds a QK matmul
    (``aten.bmm.dtype``) feeding the online-softmax chain
    (``amax -> maximum -> sub -> exp2 -> sum``) and a ``baddbmm.default``
    (P@V) whose acc is the loop-carried output.

    Returns ``(bm, bn, score_plan)`` derived from the operand shapes -- bm =
    tile_m (query rows), bn = tile_n (key tile), score_plan = tile-local
    QK-score transforms plus head_dim -- or None when the loop is not the
    attention pattern or the validated config envelope is not met.

    This is deliberately strict so single-GEMM detection
    (``_detect_specialized_mma_loop``) is 100% untouched.
    """
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    # Attention binds to a 2-axis root grid (tile_b, tile_m) plus one inner
    # tile_n device loop.
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 2:
        return None
    if len(block_ids) != 1 or any(bid in root_grid_ids for bid in block_ids):
        return None

    graph_info = None
    for candidate in fn.codegen.codegen_graphs:
        if isinstance(candidate, ForLoopGraphInfo) and candidate.block_ids == block_ids:
            graph_info = candidate
            break
    if graph_info is None:
        return None

    pattern = _attention_softmax_pattern_head_dim(
        graph_info.graph,
        kv_block_id=block_ids[0],
    )
    if pattern is None:
        return None

    env = CompileEnvironment.current()

    # The flash kernel walks full 128-row/128-col tiles, so BOTH the query
    # (tile_m) and key/value (tile_n) sequence lengths must be exact multiples of
    # 128. ``known_multiple`` is conservative under dynamic shapes (False unless
    # PROVABLY divisible), so a runtime-variable seq keeps the scalar path. This
    # mirrors the codegen's seq%128 gate at detection time, where the device
    # function arguments are not yet populated -- so the detector never sets
    # ``mma_mode`` / forces ``block=(128,1,1)`` on a config codegen would reject.
    if not env.block_sizes[root_grid_ids[1]].known_multiple(128):
        return None
    if not env.block_sizes[block_ids[0]].known_multiple(128):
        return None
    # The flash codegen derives num_kv from the QUERY length, so it only handles
    # SQUARE self-attention (query seq == key/value seq). Cross-attention
    # (q seq != kv seq, e.g. the 1024-query/512-kv test_attention_block_pointer
    # shape) would iterate the wrong number of KV tiles -> keep the scalar path.
    # Require statically-equal concrete sizes: a dynamic (SymInt) seq is rejected
    # conservatively (size hints can collide for distinct unbacked symbols).
    q_seq = env.block_sizes[root_grid_ids[1]].size
    kv_seq = env.block_sizes[block_ids[0]].size
    if not (isinstance(q_seq, int) and isinstance(kv_seq, int) and q_seq == kv_seq):
        return None
    # The flash kernel emits its OWN (batch*head, query-tile) program mapping; it
    # is only valid for the default flat pid with no L2 reordering. Persistent /
    # interleaved pids and l2_grouping remap program ids, so the emitted tile
    # indexing would address the wrong (batch, head, query-tile) -> reject.
    if config.pid_type != "flat":
        return None
    if any(grouping != 1 for grouping in config.l2_groupings):
        return None
    if any(order != [*range(len(order))] for order in config.loop_orders):
        return None
    if any(thread_count != 0 for thread_count in config.num_threads):
        return None
    cute_vector_widths = config.config.get("cute_vector_widths", [])
    if isinstance(cute_vector_widths, list) and any(
        width != 1 for width in cute_vector_widths
    ):
        return None

    # bm/bn from the config block sizes for tile_m (root grid axis 1) and tile_n
    # (the device loop block), NOT from the grid sizes.
    bm = env.block_sizes[root_grid_ids[1]].from_config(config)
    (bn,) = (env.block_sizes[bid].from_config(config) for bid in block_ids)
    if not isinstance(bm, int) or not isinstance(bn, int):
        return None
    # tile_b must be one (batch, head) per CTA -- the flash kernel maps the
    # b*h flatten one row per program. A tile_b block > 1 would pack multiple
    # (batch, head) rows per CTA and miscompute. Enforce what the docstrings
    # claim (previously unchecked).
    tile_b = env.block_sizes[root_grid_ids[0]].from_config(config)
    if not isinstance(tile_b, int) or tile_b != 1:
        return None
    return bm, bn, pattern.score_plan


def _detect_attention_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """True when the tile_n loop is a fused tcgen05 flash-attention body.

    Gated behind ``HELION_CUTE_FLASH`` and the validated config envelope
    (tile_b block = 1, mma_tiler 128x128, head_dim in {64, 128}, fp16/bf16) so the
    default scalar-fallback path is unchanged while the path is incomplete.
    """
    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return False
    shape = _attention_loop_shape(fn, block_ids, config=config)
    if shape is None:
        return False
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    root_block_ids = device_ir.grid_block_ids[0]
    bm, bn, score_plan = shape
    from .cute.cute_flash import flash_attention_graph_lse_plan_valid

    if not flash_attention_graph_lse_plan_valid(
        fn,
        root_block_ids=root_block_ids,
        kv_block_id=block_ids[0],
        score_plan=score_plan,
    ):
        return False
    bk = score_plan.head_dim
    # Validated envelope mirrors the spike: 128x128 mma_tiler, head_dim 64/128.
    if bm == 128 and bn == 128 and bk in (64, 128):
        fn.cute_state.attention_flash_score_plan = score_plan
        return True
    return False


def _is_mma_candidate_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
    grid_ids: set[int],
) -> bool:
    if not any(bid not in grid_ids for bid in block_ids):
        return False
    resolved_threads: list[int] = [
        num_threads
        if num_threads > 0
        else int(block_size)
        if isinstance(block_size, int)
        else 0
        for block_size, num_threads in zip(block_sizes, num_threads_config, strict=True)
    ]
    return _detect_mma_loop(
        fn,
        block_ids,
        block_sizes=block_sizes,
        num_threads_config=resolved_threads,
    )


def _loop_may_use_mma(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    from ..language._decorators import is_api_func
    from .cute.cute_mma import can_codegen_cute_mma_aten
    from .cute.cute_mma import can_codegen_cute_mma_dot
    from .device_ir import RootGraphInfo
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    graph_by_id = {
        graph_info.graph_id: graph_info
        for graph_info in fn.codegen.codegen_graphs
        if hasattr(graph_info, "graph")
    }

    def graph_contains_mma(graph: object) -> bool:
        if not isinstance(graph, torch.fx.Graph):
            return False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                return True
            if is_api_func(node.target) and getattr(node.target, "__name__", "") in {
                "_for_loop",
                "_for_loop_step",
            }:
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int):
                    nested = graph_by_id.get(graph_id)
                    if nested is not None and graph_contains_mma(nested.graph):
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
        if graph_contains_mma(getattr(graph_info, "graph", None)):
            return True
    return False


def _kernel_has_specialized_mma_path(
    fn: DeviceFunction,
    *,
    config: Config,
) -> bool:
    from .compile_environment import CompileEnvironment
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
        block_sizes: list[int | torch.SymInt] = []
        for bid in block_ids:
            size = env.block_sizes[bid].from_config(config)
            if not isinstance(size, (int, torch.SymInt)):
                break
            block_sizes.append(size)
        else:
            if _detect_specialized_mma_loop(
                fn,
                list(block_ids),
                block_sizes=block_sizes,
                config=config,
            ):
                return True
    return False


def _kernel_specialized_mma_impl(
    fn: DeviceFunction,
    *,
    config: Config,
) -> str | None:
    from ..language._decorators import is_api_func
    from .compile_environment import CompileEnvironment
    from .cute.cute_mma import _choose_mma_impl
    from .cute.cute_mma import can_codegen_cute_mma_aten
    from .cute.cute_mma import can_codegen_cute_mma_dot
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
        host_device_ir = HostFunction.current().device_ir
        if len(host_device_ir.grid_block_ids) != 1:
            continue
        root_grid_ids = host_device_ir.grid_block_ids[0]
        if len(root_grid_ids) != 2:
            continue
        bm = env.block_sizes[root_grid_ids[0]].from_config(config)
        bn = env.block_sizes[root_grid_ids[1]].from_config(config)
        if not isinstance(bm, int) or not isinstance(bn, int):
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                lhs_node = node.args[1]
            elif (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                lhs_node = node.args[0]
            else:
                continue
            if not isinstance(lhs_node, torch.fx.Node):
                continue
            lhs_val = lhs_node.meta.get("val")
            if not isinstance(lhs_val, torch.Tensor):
                continue
            mma_impl = _choose_mma_impl(
                lhs_val.dtype, bm=bm, bn=bn, bk=bk, config=config
            )
            if mma_impl != "universal":
                return mma_impl
    return None


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


def _loop_contains_atomic(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    from ..language._decorators import is_api_func
    from ..language.atomic_ops import ATOMIC_OPS as atomic_targets
    from .device_ir import RootGraphInfo
    from .host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    graph_by_id = {
        graph_info.graph_id: graph_info
        for graph_info in fn.codegen.codegen_graphs
        if hasattr(graph_info, "graph")
    }

    def graph_contains_atomic(graph: object) -> bool:
        if not isinstance(graph, torch.fx.Graph):
            return False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in atomic_targets:
                return True
            if is_api_func(node.target) and getattr(node.target, "__name__", "") in {
                "_for_loop",
                "_for_loop_step",
            }:
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int):
                    nested = graph_by_id.get(graph_id)
                    if nested is not None and graph_contains_atomic(nested.graph):
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
        if graph_contains_atomic(getattr(graph_info, "graph", None)):
            return True
    return False


def _graph_used_block_ids(
    fn: DeviceFunction,
    block_ids: list[int],
) -> set[int]:
    from .compile_environment import CompileEnvironment
    from .device_ir import RootGraphInfo
    from .host_function import HostFunction

    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    candidate_block_ids = set(block_ids)
    used: set[int] = set()

    def visit_value(value: object) -> None:
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                visit_value(dim)
            return
        if isinstance(value, torch.SymInt):
            block_id = env.get_block_id(value)
            if block_id is not None and block_id in candidate_block_ids:
                used.add(block_id)
            raw_expr = getattr(getattr(value, "node", None), "_expr", None)
            if isinstance(raw_expr, sympy.Expr):
                visit_value(raw_expr)
            return
        if isinstance(value, sympy.Expr):
            for symbol in value.free_symbols:
                block_id = env.get_block_id(symbol)
                if block_id is not None and block_id in candidate_block_ids:
                    used.add(block_id)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit_value(key)
                visit_value(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit_value(item)

    def is_tensor_like_value(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(
                is_tensor_like_value(key) or is_tensor_like_value(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(is_tensor_like_value(item) for item in value)
        return False

    for block_id in candidate_block_ids:
        block_info = env.block_sizes[block_id]
        if block_info.reduction:
            used.add(block_id)

    def graph_matches_loop(graph_info: object) -> bool:
        if getattr(graph_info, "block_ids", None) == block_ids:
            return True
        if not isinstance(graph_info, RootGraphInfo):
            return False
        try:
            phase_index = device_ir.root_ids.index(graph_info.graph_id)
        except ValueError:
            return False
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    for graph_info in fn.codegen.codegen_graphs:
        if not graph_matches_loop(graph_info):
            continue
        graph = getattr(graph_info, "graph", None)
        if graph is None:
            continue
        for node in graph.nodes:
            value = node.meta.get("val")
            if is_tensor_like_value(value):
                visit_value(value)
                for arg in node.args:
                    if is_tensor_like_value(arg):
                        visit_value(arg)
                for arg in node.kwargs.values():
                    if is_tensor_like_value(arg):
                        visit_value(arg)
    return used


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


# Leave broad headroom below the G1 sweep's 3600s subprocess timeout: budget
# checks happen between inline CuTe compile/benchmark units, then the selected
# config still has to compile, pass correctness, and run the final benchmark.
_CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS = 600


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

    def validate_environment(self) -> None:
        from .cute.cutedsl_compat import check_cute_backend_requirements

        check_cute_backend_requirements()

    def config_value_priors(self, config_spec: ConfigSpec) -> dict[str, ValuePrior]:
        """Bias the random half of the initial population toward the config
        family that performs well on Blackwell tcgen05 kernels.

        This encodes, as a distribution, what the backend's former hardcoded
        per-shape seed configs all converged on: a 2-CTA, TMA-fed,
        role-local-monolithic, static-persistent, tvm-ffi launch with deep AB
        staging and a 4-warp epilogue. Keys a given kernel does not expose
        (e.g. the ``tcgen05_*`` keys on a pointwise/reduction kernel) are
        ignored, and values a fragment cannot represent are dropped, so the
        priors are safe for any cute kernel -- a non-matmul kernel just picks up
        the generic biases (TMA indexing, 8 warps) on whichever keys it has.
        """
        from ..autotuner.config_priors import weighted_choice
        from .cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
        from .cute.strategies import TCGEN05_STRATEGY_CONFIG_KEY
        from .cute.strategies import Tcgen05PersistenceModel
        from .cute.strategies import Tcgen05Strategy
        from .cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE

        priors: dict[str, ValuePrior] = {
            # Generic knobs shared by every cute kernel.
            "num_warps": weighted_choice({8: 4.0, 4: 2.0, 16: 1.0}),
            "num_stages": weighted_choice({4: 3.0, 3: 2.0, 2: 1.0}),
            "indexing": weighted_choice(
                {"tensor_descriptor": 4.0, "pointer": 1.0, "block_ptr": 1.0}
            ),
            "pid_type": weighted_choice(
                {
                    TCGEN05_TWO_CTA_SEED_PID_TYPE: 3.0,
                    "flat": 1.0,
                    "persistent_blocked": 1.0,
                }
            ),
            # tcgen05 / 2-CTA matmul knobs (absent on non-matmul kernels).
            "tcgen05_cluster_m": weighted_choice({2: 3.0, 1: 1.0}),
            "tcgen05_ab_stages": weighted_choice(
                {3: 3.0, 4: 2.0, 5: 1.0, 6: 1.0, 2: 1.0}
            ),
            "tcgen05_acc_stages": weighted_choice({2: 4.0, 1: 1.0}),
            "tcgen05_c_stages": weighted_choice({2: 3.0, 4: 2.0, 1: 1.0}),
            "tcgen05_num_epi_warps": weighted_choice({4: 3.0, 2: 1.0, 8: 1.0}),
            TCGEN05_STRATEGY_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value: 3.0,
                    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value: 1.0,
                }
            ),
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value: 3.0,
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value: 1.0,
                }
            ),
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: weighted_choice({True: 3.0, False: 1.0}),
        }
        if config_spec is not None and config_spec.cute_flash_search_enabled:
            priors.update(self._cute_flash_config_value_priors(config_spec))
        return priors

    @staticmethod
    def _cute_flash_config_value_priors(
        config_spec: ConfigSpec,
    ) -> dict[str, ValuePrior]:
        from ..autotuner.config_priors import weighted_choice
        from .cute.cute_flash import flash_attention_value_prior_weights

        return {
            key: weighted_choice(weights)
            for key, weights in flash_attention_value_prior_weights(
                config_spec._cute_flash_head_dim or 0,
                config_spec._cute_flash_num_kv,
                is_causal=config_spec._cute_flash_is_causal,
                has_kv_tile_pruning=config_spec._cute_flash_has_kv_tile_pruning,
                requires_ws_overlap=config_spec._cute_flash_requires_ws_overlap,
            ).items()
        }

    def customize_ast(self, hf: HostFunction) -> None:
        """CuTe-specific AST rewrites that rewrite high-level patterns into
        equivalent forms that compile to materially faster code.

        Currently:
          * ``rewrite_online_to_3pass`` rewrites the online two-pass
            softmax pattern into the 3-pass form (max-only, then
            sum-only, then consume).  The 3-pass form's two reductions
            are independent and compile to a more efficient layout on
            the CuTe backend.
        """
        from .cute.online_to_3pass import rewrite_online_to_3pass

        rewrite_online_to_3pass(hf)

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .cute.layout_propagation import plan_layouts
        from .cute.view_subtile import annotate_view_subtiles

        annotate_view_subtiles(graphs, config)
        plan_layouts(graphs, config, tile_strategy)

    def supports_config_key(self, key: str) -> bool:
        if (
            key == "num_threads"
            or key == "cute_vector_widths"
            or key.startswith(("tcgen05_", "cute_flash_"))
        ):
            return True
        return super().supports_config_key(key)

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        if (
            inductor_dtype := CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        ) is not None:
            return inductor_dtype
        if dtype is torch.float4_e2m1fn_x2:
            # PyTorch's shell dtype stores two E2M1 values in one byte.  CuTe
            # does not support scalar dereference for its 4-bit type yet, so
            # SIMT scalar loads treat the tensor as raw byte storage.
            return "cutlass.Uint8"
        if dtype is torch.uint64:
            return "cutlass.Int64"

        raise ValueError(f"Unsupported dtype for Cute backend: {dtype}")

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    def supports_precompile(self) -> bool:
        # The CuTe DSL does not expose a Triton-style precompile entry point;
        # the autotuner has to compile + benchmark each config inline.
        return False

    def setup_compile_cache_dir(self, device_index: int) -> None:
        if "CUTE_DSL_CACHE_DIR" not in os.environ:
            from ..autotuner.local_cache import helion_cute_cache_dir

            cute_dir = helion_cute_cache_dir(device_index)
            os.environ["CUTE_DSL_CACHE_DIR"] = cute_dir
            log.debug("Set CUTE_DSL_CACHE_DIR=%s", cute_dir)

    def make_ephemeral_cache(
        self,
    ) -> contextlib.AbstractContextManager[None] | None:
        if self.keep_compile_cache_requested():
            return None
        return self._ephemeral_cute_cache()

    @contextlib.contextmanager
    def _ephemeral_cute_cache(self) -> Generator[None, None, None]:
        """Redirect the CuTe DSL on-disk cache to a temporary dir during
        autotuning so candidate compilations don't pollute the real cache.

        The winning config's artifact is re-persisted from memory into the
        real cache afterward (see :meth:`finalize_ephemeral_cache`).
        """
        saved = os.environ.get("CUTE_DSL_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="helion_cute_autotune_") as ephemeral:
            os.environ["CUTE_DSL_CACHE_DIR"] = ephemeral
            log.debug("Ephemeral CuTe cache: %s", ephemeral)
            try:
                yield
            finally:
                if saved is not None:
                    os.environ["CUTE_DSL_CACHE_DIR"] = saved
                else:
                    os.environ.pop("CUTE_DSL_CACHE_DIR", None)

    def finalize_ephemeral_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        """Persist the winning config's compiled artifact into the real cache.

        Candidate artifacts died with the ephemeral dir, but the winner's
        launcher still holds the compiled module in memory and the disk-cache
        key excludes ``CUTE_DSL_CACHE_DIR``, so re-persisting from memory
        writes the exact artifact a later process will look up.  Launchers and
        compile-cache entries are kept so the winner launches without
        recompiling.
        """
        from ..runtime.config import Config

        compiled_fn = bound_kernel._compile_cache.get(config)
        if compiled_fn is None:
            # The autotuner may return a minimized config (default values
            # stripped); the compiled entry is keyed by the full config.
            default = bound_kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            full_config = Config(**(default.config | config.config))
            compiled_fn = bound_kernel._compile_cache.get(full_config)
        if compiled_fn is None:
            return
        cute_kernel = compiled_fn.__globals__.get(  # type: ignore[attr-defined]
            f"_helion_{bound_kernel.kernel.name}"
        )
        launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", None)
        if not launchers:
            return
        device_index = (
            bound_kernel.env.device.index
            if bound_kernel.env.device.index is not None
            else 0
        )
        # The ephemeral context restored CUTE_DSL_CACHE_DIR on exit; this sets
        # the real per-device dir when the user did not provide one.
        self.setup_compile_cache_dir(device_index)
        for launcher in launchers.values():
            launcher.persist_compiled()

    def compiled_cache_key(
        self, bound_kernel: BoundKernel[Any], compiled_fn: object
    ) -> str | None:
        cute_kernel = compiled_fn.__globals__.get(  # type: ignore[attr-defined]
            f"_helion_{bound_kernel.kernel.name}"
        )
        if cute_kernel is None:
            return None
        launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", None)
        if not launchers:
            return None
        for launcher in launchers.values():
            key = getattr(launcher, "_cache_key", None)
            if key is not None:
                return key
        return None

    def annotate_compiled_module(
        self, module: object, source: str, kernel_name: str
    ) -> None:
        cute_kernel = getattr(module, f"_helion_{kernel_name}", None)
        if cute_kernel is None:
            return
        with contextlib.suppress(AttributeError, TypeError):
            cute_kernel._helion_cute_source_hash = hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest()

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # Exceptions raised from inside the cute/cutlass DSL during compile or
        # launch are expected when an invalid config is tried; treat them as
        # benign so the autotuner moves on. Anything else (AssertionError,
        # NameError/TypeError/AttributeError from generated code, ...) is
        # almost certainly a real Helion bug and is surfaced at warn level so
        # it gets noticed without terminating tuning.
        cls = type(err)
        module = getattr(cls, "__module__", "") or ""
        if module.startswith(("cutlass", "cute")):
            return "debug"
        if isinstance(err, Exception):
            return "warn"
        return None

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        # The default Triton do_bench uses CUDA events that mis-time the CuTe
        # path on Blackwell - launches show up as ~5ms when the kernel is
        # actually 250ms+. That happens because Triton records events on its
        # own driver device-interface stream, not the stream CuTe launches on.
        # Recording torch.cuda.Events on the current torch stream (the CuTe
        # launch stream) restores correct device-time timing while excluding
        # CPU launch overhead; opt in via HELION_AUTOTUNE_CUTE_CUDA_EVENTS.
        # Otherwise fall back to synchronized wall-clock timing.
        from ..runtime.settings import _env_get_bool

        if _env_get_bool("HELION_AUTOTUNE_CUTE_CUDA_EVENTS", False):
            from ..autotuner.benchmarking import do_bench_cuda_events

            return do_bench_cuda_events

        from ..autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        # Same rationale as get_do_bench: the default interleaved bench uses
        # CUDA events that mis-time the CuTe path. Use the synchronized
        # wall-clock fallback so the autotuner's interleaved compare path
        # produces real timings.
        from ..autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        original_budget = bound_kernel.settings.autotune_budget_seconds
        # "full" should mean the search algorithms run to completion unless the
        # caller explicitly provides a budget. Keep the defensive default budget
        # only for cheaper efforts where bounded local iteration is expected.
        if (
            bound_kernel.settings.autotune_budget_seconds is None
            and bound_kernel.settings.autotune_effort != "full"
        ):
            bound_kernel.settings.autotune_budget_seconds = (
                _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS
            )
        try:
            return super().autotune(bound_kernel, args, force=force, **kwargs)
        finally:
            bound_kernel.settings.autotune_budget_seconds = original_budget

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        return "cutlass.Constexpr"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    @property
    def default_launcher_name(self) -> str:
        return "_default_cute_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "operator": "import operator",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "cutlass": "import cutlass",
            "cute": "import cutlass.cute as cute",
            "ir": "from cutlass._mlir import ir",
            "mlir_math": "from cutlass._mlir.dialects import math as mlir_math",
            "_default_cute_launcher": "from helion.runtime import default_cute_launcher as _default_cute_launcher",
            "_next_power_of_2": "from helion._utils import next_power_of_2 as _next_power_of_2",
            "_cute_argreduce_index": "from helion._compiler.cute.reduce_helpers import _cute_argreduce_index",
            "_helion_tcgen05_pipeline": (
                "from helion._compiler.cute import tcgen05_pipeline "
                "as _helion_tcgen05_pipeline"
            ),
            "_cute_gelu_erf_exact_f32x2": (
                "from helion._compiler.cute.epilogue_helpers import "
                "gelu_erf_exact_f32x2 as _cute_gelu_erf_exact_f32x2"
            ),
            "_cute_grouped_reduce_shared_tree": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_tree",
            "_cute_grouped_reduce_shared_two_stage": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_two_stage",
            "_cute_grouped_reduce_warp": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_warp",
            "_cute_pre_vec_fold": "from helion._compiler.cute.reduce_helpers import _cute_pre_vec_fold",
            "_cute_store_shared_remote_x4": "from helion._compiler.cute.cluster_helpers import store_shared_remote_x4 as _cute_store_shared_remote_x4",
            "_cute_issue_clc_query_nomulticast": "from helion._compiler.cute.clc_helpers import issue_clc_query_nomulticast as _cute_issue_clc_query_nomulticast",
            "_cute_inline_asm_elementwise": "from helion._compiler.cute.inline_asm_helpers import inline_asm_elementwise as _cute_inline_asm_elementwise",
            "_cute_fp8e4m3fn_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_to_float32 as _cute_fp8e4m3fn_to_float32",
            "_cute_fp8e4m3fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_x2_to_float32 as _cute_fp8e4m3fn_x2_to_float32",
            "_cute_float4_e2m1fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import float4_e2m1fn_x2_to_float32 as _cute_float4_e2m1fn_x2_to_float32",
            "_cute_grid_barrier": "from helion._compiler.cute.grid_barrier import grid_barrier as _cute_grid_barrier",
            "_cute_atomic_max_float32": "from helion._compiler.cute.atomic_helpers import atomic_max_float32 as _cute_atomic_max_float32",
            "_cute_atomic_min_float32": "from helion._compiler.cute.atomic_helpers import atomic_min_float32 as _cute_atomic_min_float32",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"{index_dtype}(cute.arch.block_idx()[{dim}])"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLArg
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        class HelionCuteDSLOpOverrides(CuteDSLOpOverrides):
            @staticmethod
            def where(
                condition: CuteDSLArg,
                a: CuteDSLArg,
                b: CuteDSLArg,
            ) -> CuteDSLArg:
                tensor_arg = (
                    HelionCuteDSLOpOverrides._get_cse_var(a)
                    or HelionCuteDSLOpOverrides._get_cse_var(b)
                    or HelionCuteDSLOpOverrides._get_cse_var(condition)
                )
                if tensor_arg is not None:
                    return CuteDSLOpOverrides.where(condition, a, b)
                return f"(({a}) if ({condition}) else ({b}))"

        return HelionCuteDSLOpOverrides()

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{dtype_str}({expr_str})"

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        from .device_function import DeviceFunction
        from .device_function import NoCurrentFunction

        if isinstance(x, ast.Name):
            try:
                df = DeviceFunction.current()
            except NoCurrentFunction:
                pass
            else:
                if (
                    df.cute_state.get_tcgen05_store_value(df.variable_aliases(x.id))
                    is not None
                ):
                    return x
        return super().cast_ast(x, target_dtype)

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        # ``sem_arg`` is a TensorArg that arrives as a ``cute.Tensor``; its
        # ``.iterator`` is the underlying ``cute.Pointer`` to the semaphore.
        return f"_cute_grid_barrier({sem_arg}.iterator)"

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        from .compile_environment import CompileEnvironment

        index_dtype = CompileEnvironment.current().index_type()
        return (
            f"{offset_var} + {index_dtype}(cute.arch.thread_idx()[{axis}])"
            f" * {elements_per_thread}"
        )

    def lane_offset_expr(self, lane_var: str) -> str:
        return f"cutlass.Int32({lane_var})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .device_function import cute_texpr

        return cute_texpr(expr)

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(f"cutlass.Int32({begin})")
        range_args.append(f"cutlass.Int32({end})")
        if step is not None and step != "1":
            range_args.append(f"cutlass.Int32({step})")
        return f"range({', '.join(range_args)})"

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
            f"{offsets_var} = ({lid}) * ({block_size_var})"
            f" + {dtype}(cute.arch.thread_idx()[{axis}])"
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if axis >= 3 and block_size_var != "1":
            raise exc.BackendUnsupported(self.name, f"thread axis {axis}")
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {dtype}(cute.arch.thread_idx()[{axis}])"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return self.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def max_reduction_threads(self) -> int | None:
        return 1024

    def max_reduction_loop(self) -> int | None:
        from .reduction_strategy import cute_looped_reduction_block_size

        max_threads = self.max_reduction_threads()
        if max_threads is None:
            return None
        return cute_looped_reduction_block_size(2**31 - 1, max_threads)

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        from .cute.thread_budget import MAX_THREADS_PER_BLOCK
        from .reduction_strategy import ReductionStrategy

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
        while other_threads * requested > MAX_THREADS_PER_BLOCK and requested > 1:
            requested //= 2
        return requested

    def create_synthetic_reduction_lanes(
        self,
        thread_count: int,
        size_hint: int,
    ) -> int | None:
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        if thread_count <= 0:
            return None
        padded_size = next_power_of_2(max(1, size_hint))
        if padded_size > thread_count:
            return padded_size // thread_count
        return None

    def reduction_axis_first(self) -> bool:
        return True

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        return True

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # One element per thread: tile-shaped temporaries are scalars.
        return f"{self.dtype_str(dtype)}({value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"({true_val}) if ({mask}) else ({false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        from .compile_environment import CompileEnvironment

        index_type = CompileEnvironment.current().index_type()
        lhs = self.cast_expr(a, index_type)
        rhs = self.cast_expr(b, index_type)
        return f"({lhs}) if ({lhs}) < ({rhs}) else ({rhs})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "cutlass.Int32(0)"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"_next_power_of_2({expr})"

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        # Use Python ternary instead of cute.where for max/min because
        # these operate on scalar registers, not tensors.
        #
        # Cast the incoming value to the accumulator dtype first.  The
        # accumulator is promoted to the computation dtype (fp32 for
        # half-precision inputs), but the per-iteration reduction input keeps
        # the tensor's storage dtype (e.g. bf16 for a masked half load).  The
        # CUTLASS DSL strictly type-checks the two branches of a Python ternary
        # ("Then and else blocks of ifexp return different types"), so a bare
        # ``acc if acc > val else val`` with mixed fp32/bf16 operands fails to
        # compile.  The cast is a no-op when ``val`` already matches.
        val = self.cast_expr(val, self.dtype_str(dtype))
        if reduction_type == "sum":
            return f"({acc} + {val})"
        if reduction_type == "max":
            return f"({acc}) if ({acc}) > ({val}) else ({val})"
        if reduction_type == "min":
            return f"({acc}) if ({acc}) < ({val}) else ({val})"
        if reduction_type == "prod":
            return f"({acc} * {val})"
        raise exc.BackendUnsupported(self.name, f"reduction combine {reduction_type!r}")

    def _threads_for_block_size_var(self, block_size_var: str | None) -> int:
        # threads_in_group must be a Python int literal for CuTe DSL.
        from .reduction_strategy import ReductionStrategy
        from .tile_strategy import BlockSizeTileStrategy

        threads = 32
        strategies = self._get_strategies()
        if block_size_var is not None:
            for strategy in strategies:
                if not isinstance(strategy, ReductionStrategy):
                    continue
                strategy_bs_var = strategy.block_size_var(strategy.block_index)
                if strategy_bs_var != block_size_var:
                    continue
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc

            # Block reductions are keyed by a tile block-size var rather than a
            # ReductionStrategy var. Recover the tile width from the owning strategy.
            for strategy in strategies:
                if not isinstance(strategy, BlockSizeTileStrategy):
                    continue
                for idx, block_id in enumerate(strategy.block_ids):
                    strategy_bs_var = strategy.block_size_var(block_id)
                    if strategy_bs_var != block_size_var:
                        continue
                    block_size = strategy.block_size
                    if isinstance(block_size, list) and idx < len(block_size):
                        block_size = block_size[idx]
                    if isinstance(block_size, int) and block_size > 0:
                        return min(block_size, 32)
            return threads

        for strategy in strategies:
            if isinstance(strategy, ReductionStrategy):
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc
        return threads

    def reduction_threads_hint(self, block_size_var: str | None = None) -> int | None:
        return self._threads_for_block_size_var(block_size_var)

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        threads = (
            threads_in_group
            if threads_in_group is not None
            else self._threads_for_block_size_var(block_size_var)
        )
        tg = f", threads_in_group={threads}"
        if reduction_type == "sum":
            return f"cute.arch.warp_reduction_sum({input_name}{tg})"
        if reduction_type == "max":
            return f"cute.arch.warp_reduction_max({input_name}{tg})"
        if reduction_type == "min":
            return (
                f"cute.arch.warp_reduction("
                f"{input_name}, lambda a, b: (a if a < b else b){tg})"
            )
        if reduction_type == "prod":
            return f"cute.arch.warp_reduction({input_name}, lambda a, b: (a * b){tg})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def thread_linear_index_expr(self, axis_sizes: dict[int, int]) -> str | None:
        from .compile_environment import CompileEnvironment

        index_dtype = CompileEnvironment.current().index_dtype
        index_type = self.index_type_str(index_dtype)
        if not axis_sizes:
            return self.cast_expr("0", index_type)
        stride = 1
        terms: list[str] = []
        for axis, size in sorted(axis_sizes.items()):
            term = self.cast_expr(f"cute.arch.thread_idx()[{axis}]", index_type)
            if stride != 1:
                term = f"({term}) * {self.cast_expr(repr(stride), index_type)}"
            terms.append(term)
            stride *= size
        return " + ".join(terms)

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
    ) -> str:
        if index_dtype is None:
            raise exc.BackendUnsupported(self.name, "missing index_dtype for argreduce")
        value_reduction = "min" if reduction_type == "argmin" else "max"
        reduced_value = self.reduction_expr(
            input_name,
            value_reduction,
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        index_dtype_str = self.index_type_str(index_dtype)
        max_index = self.cast_expr(repr(torch.iinfo(index_dtype).max), index_dtype_str)
        candidate_index = f"({index_value}) if (({input_name}) == ({reduced_value})) else ({max_index})"
        reduced_index = self.reduction_expr(
            candidate_index,
            "min",
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        return self.cast_expr(reduced_index, self.dtype_str(output_dtype))

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        if reduction_type == "argmin":
            better = (
                f"(({value}) < ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        else:
            better = (
                f"(({value}) > ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        return [
            (
                f"{acc}, {acc_index} = "
                f"(({value}), ({index})) if ({better}) else (({acc}), ({acc_index}))"
            )
        ]

    def _get_strategies(self) -> list[TileStrategy]:
        """Get the current device function's strategies."""
        from .device_function import DeviceFunction

        try:
            return DeviceFunction.current().tile_strategy.strategies
        except Exception:
            return []

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from .cute.thread_budget import MAX_THREADS_PER_BLOCK
        from .device_function import DeviceFunction
        from .host_function import HostFunction

        device_function = DeviceFunction.current()
        codegen = device_function.codegen
        tile_strategy = device_function.tile_strategy
        final_kernel_text = "\n".join(
            ast.unparse(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )
        final_thread_axes = {
            int(axis_text)
            for axis_text in re.findall(
                r"cute\.arch\.thread_idx\(\)\[(\d+)\]",
                final_kernel_text,
            )
        }

        def launcher_args_with_compile_options(block_arg: str) -> list[str]:
            launcher_args = [block_arg]
            compile_options: list[str] = []
            if config.get(TCGEN05_CUBIN_LINEINFO_CONFIG_KEY) is True:
                compile_options.append("--generate-line-info")
            # ``--enable-tvm-ffi`` is emitted in codegen only when the
            # autotune flag is True so the generated code reflects which
            # configs deliberately requested FFI. The runtime
            # (``_get_compiled_cute_launcher``) unconditionally merges
            # the flag in for the generic launcher, so configs with this
            # flag False still execute with FFI enabled — that drift is
            # intentional for now and noted here for future cleanups.
            if config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True:
                if (
                    _kernel_specialized_mma_impl(
                        device_function,
                        config=device_function.config,
                    )
                    != "tcgen05"
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY}=True requires "
                        "tcgen05 CuTe lowering",
                    )
                compile_options.append("--enable-tvm-ffi")
            if compile_options:
                launcher_args.append(
                    f"cute_compile_options={' '.join(compile_options)!r}"
                )
            return launcher_args

        # Fused tcgen05 flash-attention: 128 threads (single-warpgroup Stage-3)
        # or 256 threads (Stage-4 warp-spec producer/consumer split). The custom
        # flash codegen owns the whole device body, so the SIMT thread-axis
        # heuristics below do not apply.
        if device_function.cute_state.attention_flash_block_ids is not None:
            flash_threads = device_function.cute_state.attention_flash_threads
            return launcher_args_with_compile_options(f"block=({flash_threads}, 1, 1)")

        block_size_values = {
            name: int(value)
            for name, value in re.findall(
                r"^(_BLOCK_SIZE_\d+) = (\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        }
        # Accept both the historical ``offset_<n>`` prefix (non-CuTe backends)
        # and the post-rename ``tile_offset_<n>`` prefix (CuTe backend, see the
        # CuTe DSL preprocessor counter-collision note in
        # ``tile_strategy.py``). The launch-dim recovery walks the generated
        # source to pair Helion's per-axis offsets with their thread axes; if
        # the regex misses, we fall back to ``[1, 1, 1]`` and any kernel that
        # depends on the recovery launches with too-small dims.
        offset_block_sizes = dict(
            re.findall(
                r"^\s*(?:tile_)?offset_(\d+) = .* \* (_BLOCK_SIZE_\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        )
        offset_thread_dims = [1, 1, 1]
        # When a lane loop is active for an axis the generated index expression
        # has the form
        #   ``indices_<n> = tile_offset_<n> + Int32(thread_idx()[<axis>]) * <epT> + Int32(lane_<n>)``
        # where ``<epT>`` is ``elements_per_thread`` for that axis and the
        # outer ``for lane_<n> in range(<epT>):`` covers the residual. In
        # that case the launch-time thread extent for the axis is
        # ``block_size / <epT>``, not ``block_size``. Without dividing by
        # ``<epT>`` here the launch dim ends up at ``block_size`` while the
        # generated tile arithmetic only spans ``block_size / <epT>`` threads,
        # which means ``thread_idx[axis] >= block_size / <epT>`` writes past
        # the tile and triggers ``cudaErrorIllegalAddress`` mid-search. The
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx`` is closed
        # before the ``* <epT>`` multiplier, so the closing ``)`` between
        # ``[axis]`` and ``*`` is part of the line we have to skip over.
        #
        # The launch dim along ``axis`` has to serve **every** indices line
        # that uses that axis. If two lines on the same axis emit different
        # multipliers (e.g. one block_id with ``epT=1`` and another with
        # ``epT=2``), the line with the larger ``thread_idx[axis]`` range
        # is the binding one — so we compute ``block_size // epT`` *per
        # line* and take the ``max`` across lines, rather than combining
        # multipliers across lines and dividing once. ``re.findall`` over
        # the optional-multiplier alternation cannot be made to populate
        # the multiplier group reliably (the optional ``(?:...)?`` form
        # prefers the empty match), so we scan the lines once in Python
        # to keep the per-line ``(block_size, epT)`` pair intact.
        # ``indices_line_re`` anchors on the
        # ``cutlass.Int32(cute.arch.thread_idx()[<axis>])`` form the CuTe
        # backend emits via ``lane_index_expr`` (see backend.py:2666).
        # The trailing ``\)`` after ``\]`` is the close of the
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx``; if a
        # future codegen path drops the wrapper, the regex will
        # silently fail to match and the launch dim under-dimensions
        # without a signal. ``indices_line_assert_re`` below detects
        # the "wrapper-dropped" form so we can fail loudly instead.
        indices_line_re = re.compile(
            r"^\s*indices_\d+ = (?:tile_)?offset_(\d+) \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)\[(\d+)\]\)"
            r"(?:\s*\*\s*(?:cutlass\.Int32\()?(\d+))?",
            flags=re.MULTILINE,
        )
        # Loose form: any ``indices_<n>`` line containing ``thread_idx``
        # under any wrapping. Used only for the wrapper-invariant
        # assertion below — never consulted for launch-dim values.
        indices_line_loose_re = re.compile(
            r"^\s*indices_\d+ = (?:tile_)?offset_\d+ \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)",
            flags=re.MULTILINE,
        )
        matched_lines = 0
        for line_match in indices_line_re.finditer(final_kernel_text):
            matched_lines += 1
            offset_id = line_match.group(1)
            axis_text = line_match.group(2)
            multiplier_text = line_match.group(3)
            axis = int(axis_text)
            if not (0 <= axis < len(offset_thread_dims)):
                continue
            block_name = offset_block_sizes.get(offset_id)
            block_size = block_size_values.get(block_name or "")
            if block_size is None and block_name is not None:
                try:
                    config_index = int(block_name.removeprefix("_BLOCK_SIZE_"))
                except ValueError:
                    config_index = -1
                if 0 <= config_index < len(config.block_sizes):
                    config_block_size = config.block_sizes[config_index]
                    if isinstance(config_block_size, int):
                        block_size = config_block_size
            if block_size is None:
                continue
            elements_per_thread = int(multiplier_text) if multiplier_text else 1
            if elements_per_thread <= 0:
                # ``lane_index_expr`` only emits ``* <n>`` for ``n >= 1``;
                # a ``* 0`` multiplier would mean the index expression is
                # already invalid (every thread maps to offset 0), so
                # silently falling back to ``block_size`` here would
                # mask the very class of bug this recovery exists to
                # catch. Surface it loudly.
                raise AssertionError(
                    f"launch-dim recovery: non-positive "
                    f"elements_per_thread={elements_per_thread} for axis="
                    f"{axis}"
                )
            if block_size % elements_per_thread != 0:
                # The strategy invariant ``_thread_extent_for_axis``
                # rejects non-divisible ``block_size / nt`` at
                # construction, so the regex should never see a line
                # whose ``epT`` does not divide ``block_size`` evenly.
                # If a future codegen regression breaks that invariant,
                # surface it loudly here rather than silently
                # under-dimensioning the launch.
                raise AssertionError(
                    f"launch-dim recovery: block_size={block_size} not "
                    f"divisible by elements_per_thread={elements_per_thread} "
                    f"for axis={axis}"
                )
            line_extent = block_size // elements_per_thread
            offset_thread_dims[axis] = max(offset_thread_dims[axis], line_extent)
        # Wrapper-invariant assertion: when there is at least one
        # ``indices_<n> = ... thread_idx() ...`` line and we have
        # ``_BLOCK_SIZE_<n>`` constants to consume, the strict regex
        # above must have matched at least one of them. A loose match
        # without a strict match means the codegen emitted
        # ``thread_idx[axis]`` outside the ``cutlass.Int32(...)``
        # wrapper the strict regex anchors on; in that case the
        # launch dim would silently fall back to ``[1, 1, 1]`` and
        # under-dimension every kernel that uses a thread axis.
        if (
            offset_block_sizes
            and indices_line_loose_re.search(final_kernel_text)
            and matched_lines == 0
        ):
            raise AssertionError(
                "launch-dim recovery: indices_<n> lines reference "
                "cute.arch.thread_idx() but none matched the "
                "cutlass.Int32(...)-wrapped form. The strict regex in "
                "_launcher_block_arg is anchored on the wrapper emitted "
                "by lane_index_expr; if the codegen producer was changed "
                "to drop the wrapper, update the regex to match the new "
                "form."
            )
        dims = tuple(codegen.max_thread_block_dims)
        root_live_dims = tuple(codegen.root_thread_block_dims)
        referenced_dims = tuple(codegen.referenced_thread_block_dims)
        static_dims = tile_strategy.thread_block_dims()
        dim_exprs = tile_strategy.thread_block_dim_exprs()
        static_threads = functools.reduce(operator.mul, static_dims, 1)
        dynamic_threads = functools.reduce(operator.mul, dims, 1)
        has_nested_device_loops = any(
            getattr(graph_info, "block_ids", None) is not None
            for graph_info in codegen.codegen_graphs
        )
        root_grid_dims = [1, 1, 1]
        device_ir = HostFunction.current().device_ir
        for block_ids in device_ir.grid_block_ids:
            strategy = tile_strategy.block_id_to_strategy.get(tuple(block_ids))
            if strategy is None:
                continue
            for axis, size in enumerate(strategy.thread_block_sizes()):
                if axis < len(root_grid_dims):
                    root_grid_dims[axis] = max(root_grid_dims[axis], size)
        root_static_dims = tuple(root_grid_dims)
        root_static_threads = functools.reduce(operator.mul, root_static_dims, 1)
        specialized_root_tcgen05 = (
            _kernel_specialized_mma_impl(device_function, config=device_function.config)
            == "tcgen05"
            and root_static_dims != (1, 1, 1)
            and root_static_threads <= MAX_THREADS_PER_BLOCK
        )
        tcgen05_compact_dims = (
            device_function.cute_state.block_shape if specialized_root_tcgen05 else None
        )
        if referenced_dims != (1, 1, 1):
            dims = referenced_dims
        elif has_nested_device_loops:
            dims = tuple(codegen.max_thread_block_dims)
        if functools.reduce(operator.mul, dims, 1) > MAX_THREADS_PER_BLOCK:
            if (
                root_static_dims != (1, 1, 1)
                and root_static_threads <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_static_dims
            elif static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
        recorded_dims = tuple(
            max(
                codegen.max_thread_block_dims[axis],
                root_live_dims[axis],
                referenced_dims[axis],
            )
            for axis in range(3)
        )
        if tcgen05_compact_dims is not None:
            dims = tcgen05_compact_dims
        elif specialized_root_tcgen05:
            dims = root_static_dims
        elif final_thread_axes and (
            referenced_dims != (1, 1, 1) or has_nested_device_loops
        ):
            dims = tuple(
                max(size, root_static_dims[axis], recorded_dims[axis])
                if axis in final_thread_axes
                else size
                for axis, size in enumerate(dims)
            )
            dims = tuple(
                size if axis in final_thread_axes else 1
                for axis, size in enumerate(dims)
            )
        else:
            dims = tuple(
                min(size, recorded_dims[axis]) for axis, size in enumerate(dims)
            )
        current_threads = functools.reduce(operator.mul, dims, 1)
        if current_threads > MAX_THREADS_PER_BLOCK:
            if static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
            elif (
                root_live_dims != (1, 1, 1)
                and functools.reduce(operator.mul, root_live_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_live_dims
            elif (
                referenced_dims != (1, 1, 1)
                and functools.reduce(operator.mul, referenced_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = referenced_dims
        if (
            dims != (1, 1, 1)
            and static_dims != (1, 1, 1)
            and not has_nested_device_loops
            and static_threads < dynamic_threads
            # Synthetic free-``hl.arange`` thread axes are real launch lanes that
            # the strategy's ``static_dims`` does not know about, so do not fall
            # back to ``static_dims`` (which would drop them) when they are live.
            and not codegen.cute_synthetic_arange_axis_sizes
        ):
            dims = static_dims
        if (
            dim_exprs is not None
            and dim_exprs != ("1", "1", "1")
            # ``dim_exprs`` is the strategy's static per-axis launch shape; it
            # has no entry for synthetic free-``hl.arange`` axes, so adopting it
            # wholesale would shrink those axes back to 1. Skip it when they are
            # live (the synthetic extents are already folded into ``dims``).
            and not codegen.cute_synthetic_arange_axis_sizes
        ):
            if all(expr.isdigit() for expr in dim_exprs):
                expr_dims = tuple(int(expr) for expr in dim_exprs)
                if functools.reduce(
                    operator.mul, expr_dims, 1
                ) <= MAX_THREADS_PER_BLOCK and all(
                    expr_dim <= current_dim
                    for expr_dim, current_dim in zip(expr_dims, dims, strict=True)
                ):
                    dims = expr_dims
            elif dims == (1, 1, 1):
                return launcher_args_with_compile_options(
                    f"block=({dim_exprs[0]}, {dim_exprs[1]}, {dim_exprs[2]})"
                )
        if offset_thread_dims != [1, 1, 1]:
            candidate_dims = tuple(
                starmap(max, zip(dims, offset_thread_dims, strict=True))
            )
            if (
                functools.reduce(operator.mul, candidate_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = candidate_dims
        if dims == (1, 1, 1):
            dynamic_dims = tuple(codegen.max_thread_block_dims)
            if (
                dynamic_dims != (1, 1, 1)
                and functools.reduce(operator.mul, dynamic_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = dynamic_dims
            else:
                dims = DeviceFunction.current().tile_strategy.thread_block_dims()
        # Detect the silent-truncation case: codegen has already emitted
        # thread_idx[axis] references that assume a certain per-axis
        # extent (recorded in ``referenced_thread_block_dims``), but the
        # chosen launch ``dims`` for that axis is smaller. This happens
        # when the joint requested thread count exceeds 1024 and the
        # earlier fallback paths in this function dropped axes to (1, 1, 1)
        # to fit the budget. Under-dimensioning here would leave
        # cross-thread reductions (group_span, warp_reduce) operating
        # against nonexistent lanes, silently producing wrong results
        # (e.g. softmax with M*N>1024 returning max_err on the order of
        # tens). Surface this as ``BackendUnsupported`` so the autotuner
        # skips the config and falls back to a viable one.
        #
        # Skip this check when the joint referenced thread count fits
        # within ``MAX_THREADS_PER_BLOCK``: in that case any per-axis
        # mismatch comes from a strategy that intentionally launches
        # fewer threads than the codegen "references" (e.g. an outer-
        # tile axis with a single logical lane that still appears in a
        # warp-reduction call), and the CuTe runtime degenerates the
        # reduction to the live lanes without losing data.
        #
        # Skip this check for tcgen05-specialized matmul kernels: those
        # use a custom role-warp launch shape that intentionally differs
        # from the SIMT thread-axis counts (the latter being how many
        # threads the user-visible per-element loops would expect). The
        # tcgen05 code paths know which lanes are alive on their own.
        # Also skip when the kernel has any matmul / MMA call (checks for
        # cute.gemm calls): those paths cooperate within a warp through
        # CUTLASS MMA intrinsics that don't depend on the SIMT axis layout,
        # so the strategy can intentionally launch fewer threads on a
        # reduction axis (e.g. K) than the codegen "references" through the
        # strategy's per-block thread count.
        from .cute.thread_budget import check_thread_limit

        def _emits_cute_gemm(stmt: ast.AST) -> bool:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "gemm"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "cute"
                ):
                    return True
            return False

        kernel_has_mma = any(
            _emits_cute_gemm(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )
        if (
            tcgen05_compact_dims is None
            and not specialized_root_tcgen05
            and not kernel_has_mma
        ):
            referenced_threads = functools.reduce(
                operator.mul, codegen.referenced_thread_block_dims, 1
            )
            if referenced_threads > MAX_THREADS_PER_BLOCK:
                for axis, ref_size in enumerate(codegen.referenced_thread_block_dims):
                    if ref_size > 1 and axis < len(dims) and dims[axis] < ref_size:
                        raise exc.BackendUnsupported(
                            self.name,
                            (
                                f"launch dims {tuple(dims)} under-dimension"
                                f" referenced_thread_block_dims="
                                f"{tuple(codegen.referenced_thread_block_dims)}"
                                f" (axis {axis}: launched={dims[axis]} <"
                                f" referenced={ref_size}). Codegen would access"
                                f" nonexistent threads — joint requested"
                                f" thread count {referenced_threads} >"
                                f" {MAX_THREADS_PER_BLOCK}."
                            ),
                        )

        check_thread_limit(dims[0] * dims[1] * dims[2], context=str(tuple(dims)))
        return launcher_args_with_compile_options(
            f"block=({dims[0]}, {dims[1]}, {dims[2]})"
        )

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
        if not tensor_host_args:
            raise exc.BackendUnsupported(self.name, "kernel launch without tensor args")
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from .compile_environment import CompileEnvironment
        from .device_ir import ForLoopGraphInfo
        from .device_ir import ReductionLoopGraphInfo
        from .host_function import HostFunction
        from .tile_strategy import CuteFlattenedTileStrategy
        from .tile_strategy import CuteNDTileStrategy

        env = CompileEnvironment.current()
        device_ir = HostFunction.current().device_ir
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        flattened = block_size_infos[0].is_flattened(config)
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )
        has_device_loops = any(
            isinstance(graph, ForLoopGraphInfo)
            and not isinstance(graph, ReductionLoopGraphInfo)
            for graph in fn.codegen.codegen_graphs
        )
        has_dynamic_shape = any(env.block_sizes[i].size is None for i in block_ids)
        grid_ids = {bid for ids in device_ir.grid_block_ids for bid in ids}
        num_threads_config = [
            int(env.config_spec.num_threads.config_get(config.num_threads, block_id, 0))
            for block_id in block_ids
        ]
        # Compute the total thread count across all block dimensions
        # (grid + device loops) to check against the hardware limit.
        # When it would exceed 1024, default device-loop (non-grid)
        # dimensions to 1 thread to avoid budget overflow.
        from .cute.thread_budget import MAX_THREADS_PER_BLOCK

        def _shrink_auto_thread_counts(
            nd_block_size: Sequence[object], thread_limit: int
        ) -> int:
            int_positions: list[int] = []
            int_block_sizes: dict[int, int] = {}
            for i, block_size in enumerate(nd_block_size):
                if isinstance(block_size, int):
                    int_positions.append(i)
                    int_block_sizes[i] = block_size
            resolved_threads = [
                num_threads_config[i]
                if num_threads_config[i] > 0
                else int_block_sizes[i]
                for i in int_positions
            ]
            auto_positions = {
                pos
                for pos, block_idx in enumerate(int_positions)
                if num_threads_config[block_idx] == 0
            }
            static_threads = functools.reduce(operator.mul, resolved_threads, 1)
            while static_threads > thread_limit and auto_positions:
                shrink_idx = max(
                    (pos for pos in auto_positions if resolved_threads[pos] > 1),
                    key=lambda pos: resolved_threads[pos],
                    default=None,
                )
                if shrink_idx is None:
                    break
                block_idx = int_positions[shrink_idx]
                block_size = int_block_sizes[block_idx]
                next_threads = _largest_divisor_at_most(
                    block_size, resolved_threads[shrink_idx] - 1
                )
                if next_threads == resolved_threads[shrink_idx]:
                    break
                resolved_threads[shrink_idx] = next_threads
                num_threads_config[block_idx] = next_threads
                static_threads = functools.reduce(operator.mul, resolved_threads, 1)
            return static_threads

        active_loop_block_ids = _active_loop_block_ids(fn)
        all_block_infos = [env.block_sizes[i] for i in sorted(active_loop_block_ids)]
        total_threads = 1
        for info in all_block_infos:
            if info.reduction:
                continue
            bs = info.from_config(config)
            if isinstance(bs, int):
                nt = int(
                    env.config_spec.num_threads.config_get(
                        config.num_threads, info.block_id, 0
                    )
                )
                total_threads *= nt if nt > 0 else bs
        if total_threads > MAX_THREADS_PER_BLOCK:
            for i, block_id in enumerate(block_ids):
                if num_threads_config[i] == 0 and block_id not in grid_ids:
                    num_threads_config[i] = 1
        if (
            has_device_loops
            or has_dynamic_shape
            or len(device_ir.grid_block_ids) != 1
            or (len(block_ids) > 1 and not flattened)
        ):
            known_equal = getattr(env, "known_equal", None)

            def sizes_known_equal(
                lhs: int | torch.SymInt,
                rhs: int | torch.SymInt,
            ) -> bool:
                if known_equal is not None:
                    return known_equal(lhs, rhs)
                return bool(lhs == rhs)

            nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
            original_num_threads_config = list(num_threads_config)
            mma_candidate = _is_mma_candidate_loop(
                fn,
                block_ids,
                block_sizes=nd_block_size,
                num_threads_config=original_num_threads_config,
                grid_ids=grid_ids,
            )
            should_filter_inactive_block_ids = len(block_ids) > 1
            inactive_block_ids: set[int] = set()
            if should_filter_inactive_block_ids:
                used_block_ids = _graph_used_block_ids(fn, block_ids)
                if not used_block_ids:
                    used_block_ids = set(block_ids)
                for block_id in tuple(used_block_ids):
                    block_size = env.block_sizes[block_id].size
                    if block_size is None or not isinstance(
                        block_size, (int, torch.SymInt)
                    ):
                        continue
                    for other_block_id in block_ids:
                        if other_block_id == block_id:
                            continue
                        other_size = env.block_sizes[other_block_id].size
                        if other_size is None or not isinstance(
                            other_size, (int, torch.SymInt)
                        ):
                            continue
                        if sizes_known_equal(block_size, other_size):
                            used_block_ids.add(other_block_id)
                inactive_block_ids = set(block_ids) - used_block_ids
                for i, block_id in enumerate(block_ids):
                    if block_id in inactive_block_ids:
                        num_threads_config[i] = 1
            is_device_loop = any(bid not in grid_ids for bid in block_ids)
            reduction_axis_reserve = (
                1
                if any(info.reduction for info in env.block_sizes)
                and self.reduction_axis_first()
                else 0
            )

            def uses_thread_axis_for(
                block_id: int, block_size: object, num_threads: int
            ) -> bool:
                if block_id in inactive_block_ids:
                    return False
                if num_threads > 0:
                    return num_threads > 1
                return not (isinstance(block_size, int) and block_size == 1)

            consumed_grid_axes = 0
            if is_device_loop:
                for grid_block_id in grid_ids:
                    grid_info = env.block_sizes[grid_block_id]
                    grid_block_size = grid_info.from_config(config)
                    grid_threads = int(
                        env.config_spec.num_threads.config_get(
                            config.num_threads,
                            grid_block_id,
                            0,
                        )
                    )
                    if uses_thread_axis_for(
                        grid_block_id,
                        grid_block_size,
                        grid_threads,
                    ):
                        consumed_grid_axes += 1
            available_axes = max(0, 3 - reduction_axis_reserve - consumed_grid_axes)

            def current_strategy_axes() -> int:
                return sum(
                    int(uses_thread_axis_for(block_id, block_size, threads))
                    for block_id, block_size, threads in zip(
                        block_ids,
                        nd_block_size,
                        num_threads_config,
                        strict=True,
                    )
                )

            while current_strategy_axes() > available_axes:
                candidates = [
                    i
                    for i, (block_id, block_size, threads) in enumerate(
                        zip(block_ids, nd_block_size, num_threads_config, strict=True)
                    )
                    if threads == 0
                    and uses_thread_axis_for(block_id, block_size, threads)
                    and (not is_device_loop or block_id not in grid_ids)
                ]
                if not candidates:
                    break
                num_threads_config[candidates[-1]] = 1
            thread_limit = MAX_THREADS_PER_BLOCK
            if len(block_ids) > 1 and _loop_contains_matmul(fn, block_ids):
                forced_mma_impl = os.environ.get("HELION_CUTE_MMA_IMPL", "auto")
                if (
                    mma_candidate
                    or _kernel_has_specialized_mma_path(fn, config=config)
                    or (
                        _loop_may_use_mma(fn, block_ids)
                        and not _loop_contains_atomic(fn, block_ids)
                        and forced_mma_impl.strip().lower() != "auto"
                    )
                ):
                    thread_limit = MAX_THREADS_PER_BLOCK
                else:
                    # Matmul-heavy CuTe kernels with no viable MMA path, and
                    # especially atomic-accumulating split-K loops, can be
                    # register/smem limited well before the 1024-thread hard
                    # cap. Keep those auto-threaded ND tiles within 256
                    # threads and let lane loops cover the rest.
                    thread_limit = min(thread_limit, 256)
            if should_filter_inactive_block_ids and mma_candidate:
                inactive_block_ids.clear()
                num_threads_config = [
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                    for block_id in block_ids
                ]
            # OPTION B (matmul-contraction synthetic-lane fix): when a reduction
            # block is the *contraction* (K) axis of a matmul lowered through the
            # scalar fallback, it must keep enough real hardware threads to cover
            # its full extent so the cross-warp shared-memory reduction sums the
            # whole K.  Otherwise the budget below would hand the free tile axes
            # the threads and leave K split into ``threads x synthetic-lane`` -
            # the reduction then sums only the thread lanes, never the synthetic
            # lanes, so each contracted dot product covers only a fraction of K.
            # Reserve the K block's full thread extent up front by shrinking the
            # thread limit available to the free tile axes; the contraction axis
            # then claims that budget when its reduction strategy is created and
            # the synthetic lane is pushed onto the free tile axes instead.
            reserved_contraction_threads = self._cute_matmul_contraction_thread_reserve(
                fn, block_ids
            )
            if reserved_contraction_threads > 1:
                # Budget left for the free tile axes after the contraction axis
                # has claimed its full thread extent.
                free_budget = max(1, thread_limit // reserved_contraction_threads)
                # The free axes are built across separate ``create_loop_strategy``
                # calls but their thread counts multiply, so split the budget so
                # the product over every free axis stays within ``free_budget``.
                free_axes = self._cute_free_auto_thread_axis_count(fn, config)
                per_axis_limit = free_budget
                while per_axis_limit > 1 and per_axis_limit**free_axes > free_budget:
                    per_axis_limit //= 2
                thread_limit = max(1, per_axis_limit)
            static_threads = _shrink_auto_thread_counts(nd_block_size, thread_limit)
            from .cute.thread_budget import check_thread_limit

            # Detect MMA-compatible K-loops: device loops containing
            # addmm/mm with float16/bfloat16 operands.
            mma_mode = False
            if is_device_loop:
                mma_mode = _detect_specialized_mma_loop(
                    fn,
                    block_ids,
                    block_sizes=nd_block_size,
                    config=config,
                )
                if not mma_mode and _detect_attention_mma_loop(
                    fn,
                    block_ids,
                    config=config,
                ):
                    # Fused tcgen05 flash-attention: suppress synthetic lane
                    # loops (mma_mode) and run a single 128-thread warpgroup
                    # like the spike. The dedicated flash codegen emits the
                    # whole device body; the FX-graph statement walk is bypassed.
                    mma_mode = True
                    fn.cute_state.attention_flash_block_ids = list(block_ids)
            elif (
                len(device_ir.grid_block_ids) == 1
                and block_ids == device_ir.grid_block_ids[0]
            ):
                specialized_mma_impl = _kernel_specialized_mma_impl(fn, config=config)
                if specialized_mma_impl == "tcgen05" and len(nd_block_size) == 2:
                    from .cute.cute_mma import _tcgen05_root_m_threads

                    root_m_threads = (
                        _tcgen05_root_m_threads(
                            int(nd_block_size[0]), int(nd_block_size[1])
                        )
                        if num_threads_config[0] == 0
                        and isinstance(nd_block_size[0], int)
                        and isinstance(nd_block_size[1], int)
                        else num_threads_config[0]
                    )
                    root_n_threads = (
                        min(int(nd_block_size[1]), 8)
                        if num_threads_config[1] == 0
                        and isinstance(nd_block_size[1], int)
                        else num_threads_config[1]
                    )
                    num_threads_config[0] = root_m_threads
                    num_threads_config[1] = root_n_threads
                    static_threads = root_m_threads * root_n_threads

            check_thread_limit(static_threads, context=str(tuple(nd_block_size)))
            return CuteNDTileStrategy(
                fn,
                block_ids,
                block_size=nd_block_size,
                loop_order=loop_order,
                l2_grouping=l2_grouping,
                num_threads=num_threads_config,
                mma_mode=mma_mode,
                inactive_block_ids=inactive_block_ids,
            )
        nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
        block_size = functools.reduce(operator.mul, nd_block_size)  # pyrefly: ignore[incompatible-overload-residual]
        # Resolve per-axis thread counts then flatten to a single total
        all_auto = all(nt <= 0 for nt in num_threads_config)
        flat_num_threads = functools.reduce(
            operator.mul,
            (
                nt if nt > 0 else (int(bs) if isinstance(bs, int) else 0)
                for nt, bs in zip(num_threads_config, nd_block_size, strict=True)
            ),
            1,
        )
        if (
            isinstance(block_size, int)
            and flat_num_threads > MAX_THREADS_PER_BLOCK
            and all_auto
        ):
            # Auto thread budget exceeds the 1024-per-CTA cap: fall back to a
            # lane loop (each thread owns block_size // 1024 elements).
            flat_num_threads = MAX_THREADS_PER_BLOCK
        if isinstance(block_size, int) and flat_num_threads > 0:
            from .cute.thread_budget import check_thread_limit

            check_thread_limit(flat_num_threads, context=str(block_size))
        return CuteFlattenedTileStrategy(
            fn,
            block_ids,
            block_size=block_size,
            loop_order=loop_order,
            num_threads=flat_num_threads,
        )


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
        return f"static_cast<{dtype_str}>({expr_str})"

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        return f"{offset_var} + tid[{axis}] * {elements_per_thread}"

    def lane_offset_expr(self, lane_var: str) -> str:
        return lane_var

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"tgid[{dim}]"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + tid[{axis}]"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + tid[{axis}]"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = ({lid}) * ({block_size_var}) + tid[{axis}]"

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"tid[{axis}] < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        return True

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from .metal.metal_overrides import MetalOverrides

        return MetalOverrides()

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        metal_type = self.dtype_str(dtype)
        return f"{metal_type}({value_expr})"

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
        from .device_function import SymbolArgument

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
        from .device_function import DeviceFunction

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
        """Metal loop strategy: delegate to CuTe.

        Metal and CuTe share the same scalar-thread execution model
        (one element per thread, cooperative hardware primitives for
        matmul), so they use the same CuteND/CuteFlattenedTileStrategy
        with the same thread budget management, inactive block ID
        filtering, and auto-capping logic.

        Note: CuTe's flattened path raises ``BackendUnsupported("thread
        block too large")`` when ``block_size * num_threads > 1024``
        (the ND path auto-caps via ``_shrink_auto_thread_counts`` —
        this asymmetry is a CuTe bug to be fixed in a follow-up).
        Metal inherits this behavior for now; users hitting the error
        should pick a smaller ``block_sizes`` value.
        """
        config = self._config_with_mpp_thread_budget(fn, block_ids, config)
        # pyrefly: ignore[bad-argument-type]
        return CuteBackend.create_loop_strategy(self, fn, block_ids, config)

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

        from .host_function import HostFunction

        device_ir = HostFunction.current().device_ir
        # Only adjust the loop strategy for the root grid.  MPPGraphInfo emits
        # the cooperative K-loop internally; nested/device loops should keep
        # their normal Metal/CuTe strategy.
        if not device_ir.grid_block_ids or block_ids != device_ir.grid_block_ids[0]:
            return config
        if len(block_ids) < 2:
            return config

        from ..runtime.config import Config
        from .compile_environment import CompileEnvironment
        from .cute.thread_budget import MAX_THREADS_PER_BLOCK

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
