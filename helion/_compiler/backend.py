from __future__ import annotations

import abc
import ast
import functools
import operator
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Sequence

import sympy
import torch

from .. import exc
from .ast_extension import expr_from_string

if TYPE_CHECKING:
    from torch._inductor.ops_handler import OpsHandler

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .device_function import Argument
    from .device_function import DeviceFunction
    from .device_ir import GraphInfo
    from .tile_dispatch import TileStrategyDispatch
    from .tile_strategy import TileStrategy

    InductorOpOverrides = OpsHandler[Any]


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
    def codegen_name(self) -> str:
        """Backend name used to look up registered codegen functions."""
        return self.name

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

    def supports_precompile(self) -> bool:
        """Whether this backend supports subprocess precompilation.

        Triton backends use fork/spawn to precompile kernels and detect hangs.
        Other backends (Pallas, CuTe) may not need or support this.
        """
        return True

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
            block_size = functools.reduce(
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

        if not force and bound_kernel.kernel.configs:
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


class TritonBackend(Backend):
    """Triton code generation backend."""

    @property
    def name(self) -> str:
        return "triton"

    @property
    def experimental(self) -> bool:
        return False

    def supports_config_key(self, key: str) -> bool:
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

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return f"tl.zeros([0], {dtype})"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"triton.next_power_of_2({expr})"

    @property
    def function_decorator(self) -> str:
        return "triton.jit"

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
        from torch._inductor.codegen.triton import TritonOverrides

        return TritonOverrides()

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
}


# TPU does not natively support 64-bit element types.
_PALLAS_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend for TPU."""

    @property
    def name(self) -> str:
        return "pallas"

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
            from ..runtime.settings import is_pallas_interpret

            if tensor_host_args:
                device_expr = f"{tensor_host_args[0]}.device"
            elif is_pallas_interpret():
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
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        from ..autotuner.config_spec import BlockSizeSpec
        from .compile_environment import BlockSizeInfo

        # Map block_id -> minimum dim_from_end across all tensors
        min_dim_from_end: dict[int, int] = {}
        min_tensor_ndim: dict[int, int] = {}
        if block_sizes is not None and kernel_tensor_sizes is not None:
            for shape in kernel_tensor_sizes:
                tensor_ndim = len(shape)
                for d, dim_expr in enumerate(shape):
                    for info in block_sizes:
                        if not isinstance(info, BlockSizeInfo):
                            continue
                        if info.dim_matches(
                            dim_expr  # pyrefly: ignore[bad-argument-type]
                        ):
                            dfe = tensor_ndim - 1 - d
                            if info.block_id not in min_dim_from_end:
                                min_dim_from_end[info.block_id] = dfe
                                min_tensor_ndim[info.block_id] = tensor_ndim
                            else:
                                min_dim_from_end[info.block_id] = min(
                                    min_dim_from_end[info.block_id], dfe
                                )
                                min_tensor_ndim[info.block_id] = min(
                                    min_tensor_ndim[info.block_id],
                                    tensor_ndim,
                                )

        for i, spec in enumerate(block_specs):
            if not isinstance(spec, BlockSizeSpec):
                continue
            bid = spec.block_ids[0]
            dim_from_end = min_dim_from_end.get(bid, ndim - 1 - i)
            tensor_ndim = min_tensor_ndim.get(bid, ndim)
            requirement_alignment = self._get_pallas_required_alignment(
                dim_from_end, tensor_ndim, min_element_bits
            )

            # When the tensor dim is smaller than the alignment, any
            # block_size >= tensor_dim will be capped to tensor_dim at
            # runtime (full-dim access, always valid).  Use the
            # tensor dim as the minimum so smaller but still-valid
            # block sizes are not unnecessarily excluded.
            dim_size = next_power_of_2(max(spec.size_hint, 1))

            spec.update_min(min(requirement_alignment, dim_size))

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        return {}

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        from ..autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        from ..autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

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
            if not isinstance(arg, TensorArg) or arg.fake_value.ndim == 0:
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
        from .ast_read_writes import ReadWrites
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg
        from .host_function import HostFunction

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
            mutated_params = set(ReadWrites.from_list(host_fn.body).inplace_writes) & {
                a.arg for a in host_fn.args.args
            }
            input_storages = {id(t.untyped_storage()) for t in env.input_sources}
            # Collect reads from for-loop bodies only (kernel code), excluding
            # host-level reads like ``return out``.
            # Note: Python AST counts ``out[tile] = val`` as a Load of ``out``
            # (the object must be loaded to index into it), but from Pallas's
            # perspective this is a pure write — the tensor data is not read.
            # Subtract such "false reads" by checking inplace_writes counts.
            #
            # Only tensors allocated with torch.empty/empty_like/new_empty can be
            # output-only — their initial values are undefined, so it's safe
            # to use HBM BlockSpecs.  Tensors allocated with torch.zeros_like,
            # torch.full, etc. have meaningful initial values that must be
            # preserved via VMEM BlockSpecs.
            empty_vars = _empty_allocated_vars(host_fn.body)
            kernel_reads: set[str] = set()
            for stmt in host_fn.body:
                if isinstance(stmt, ast.For):
                    body_rw = ReadWrites.from_list(stmt.body)
                    for name, read_count in body_rw.reads.items():
                        iw_count = body_rw.inplace_writes.get(name, 0)
                        if read_count > iw_count or name not in empty_vars:
                            kernel_reads.add(name)
            for i, arg in enumerate(sorted_args):
                if not isinstance(arg, TensorArg):
                    continue
                if id(arg.fake_value.untyped_storage()) not in input_storages:
                    # Tensor created inside the function body (output)
                    output_indices.append(i)
                    if arg.host_str() in kernel_reads:
                        # Also read by the kernel (e.g. broadcast result)
                        inplace_indices.append(i)
                elif arg.host_str() in mutated_params:
                    # Input tensor mutated in-place
                    output_indices.append(i)
                    inplace_indices.append(i)

        launcher_args = [*args, f"_output_indices={output_indices}"]
        launcher_args.append(f"_inplace_indices={inplace_indices}")

        if has_rng_ops:
            launcher_args.insert(-1, "_rng_seed_buffer")

        block_spec_info = self._compute_block_spec_info(sorted_args, config)
        if block_spec_info is not None:
            if has_rng_ops:
                block_spec_info.append(None)  # RNG seed buffer is untiled
            launcher_args.append(f"_block_spec_info={block_spec_info!r}")

        from .device_function import DeviceFunction

        device_fn = DeviceFunction.current()
        smem_ids = device_fn.pallas_smem_tensor_ids
        if smem_ids and sorted_args is not None:
            smem_arg_indices = [
                i
                for i, arg in enumerate(sorted_args)
                if isinstance(arg, TensorArg) and id(arg.fake_value) in smem_ids
            ]
            launcher_args.append(f"_smem_arg_indices={smem_arg_indices!r}")

        # Pass scratch shapes for pipeline/fori_loop launcher
        pallas_loop_type = config.get("pallas_loop_type", "default")
        if pallas_loop_type in ("emit_pipeline", "fori_loop"):
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

            pipeline_ids = device_fn.pallas_pipeline_tensor_ids
            if pipeline_ids and sorted_args is not None:
                pipeline_arg_indices = [
                    i
                    for i, arg in enumerate(sorted_args)
                    if isinstance(arg, TensorArg) and id(arg.fake_value) in pipeline_ids
                ]
                launcher_args.append(f"_pipeline_arg_indices={pipeline_arg_indices!r}")

        return launcher_args

    def build_launcher_name(self, config: Config) -> str:
        """Return the launcher name to use based on ``pallas_loop_type``."""
        pallas_loop_type = config.get("pallas_loop_type", "default")
        if pallas_loop_type == "emit_pipeline":
            return "_default_pallas_pipeline_launcher"
        if pallas_loop_type == "fori_loop":
            return "_default_pallas_fori_launcher"
        return self.default_launcher_name

    def get_launcher_name(self) -> str:
        """Return the launcher name based on the current config."""
        from .device_function import DeviceFunction

        try:
            device_fn = DeviceFunction.current()
            config = device_fn.config
            return self.build_launcher_name(config)
        except Exception:
            return self.default_launcher_name

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .pallas.plan_tiling import plan_tiling

        plan_tiling(graphs, config, tile_strategy)


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
    for block_id in root_grid_ids:
        block_size = env.block_sizes[block_id].from_config(config)
        if not isinstance(block_size, int):
            return False
        root_block_sizes.append(block_size)
        threads = env.config_spec.num_threads.config_get(
            config.num_threads, block_id, 0
        )
        resolved_threads = threads if threads > 0 else block_size
        if 0 < resolved_threads < block_size:
            return False

    (bk,) = block_sizes
    if not isinstance(bk, int):
        return False
    bm, bn = root_block_sizes

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
                if _choose_mma_impl(lhs_val.dtype, bm=bm, bn=bn, bk=bk) != "universal":
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
                if _choose_mma_impl(lhs_val.dtype, bm=bm, bn=bn, bk=bk) != "universal":
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


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .cute.layout_propagation import plan_layouts

        plan_layouts(graphs, config, tile_strategy)

    def supports_config_key(self, key: str) -> bool:
        if key == "num_threads":
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

        raise ValueError(f"Unsupported dtype for Cute backend: {dtype}")

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

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
            "_default_cute_launcher": "from helion.runtime import default_cute_launcher as _default_cute_launcher",
            "_next_power_of_2": "from helion._utils import next_power_of_2 as _next_power_of_2",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"cute.arch.block_idx()[{dim}]"

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

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        del sem_arg
        raise exc.BackendUnsupported(self.name, "hl.barrier()")

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
            f" + cutlass.Int32(cute.arch.thread_idx()[{axis}])"
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if axis >= 3 and block_size_var != "1":
            raise exc.BackendUnsupported(self.name, f"thread axis {axis}")
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return self.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def max_reduction_threads(self) -> int | None:
        return 32

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
        return f"({a}) if ({a}) < ({b}) else ({b})"

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
        if final_thread_axes and (
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
        ):
            dims = static_dims
        if dim_exprs is not None and dim_exprs != ("1", "1", "1"):
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
                return [f"block=({dim_exprs[0]}, {dim_exprs[1]}, {dim_exprs[2]})"]
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
        from .cute.thread_budget import check_thread_limit

        check_thread_limit(dims[0] * dims[1] * dims[2], context=str(tuple(dims)))
        return [f"block=({dims[0]}, {dims[1]}, {dims[2]})"]

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

        def _largest_divisor_at_most(size: int, limit: int) -> int:
            for divisor in range(limit, 0, -1):
                if size % divisor == 0:
                    return divisor
            return 1

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
                return lhs == rhs

            nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
            original_num_threads_config = list(num_threads_config)
            mma_candidate = _is_mma_candidate_loop(
                fn,
                block_ids,
                block_sizes=nd_block_size,
                num_threads_config=original_num_threads_config,
                grid_ids=grid_ids,
            )
            may_use_mma = _loop_may_use_mma(fn, block_ids)
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
            thread_limit = MAX_THREADS_PER_BLOCK
            if len(block_ids) > 1 and _loop_contains_matmul(fn, block_ids):
                if mma_candidate or may_use_mma:
                    thread_limit = MAX_THREADS_PER_BLOCK
                else:
                    # Matmul-heavy CuTe kernels can be register/smem limited
                    # well before the 1024-thread hard cap. Keep their
                    # auto-threaded ND tiles within 256 threads and let lane
                    # loops cover the rest.
                    thread_limit = min(thread_limit, 256)
            if should_filter_inactive_block_ids and mma_candidate:
                inactive_block_ids.clear()
                num_threads_config = [
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                    for block_id in block_ids
                ]
            static_threads = _shrink_auto_thread_counts(nd_block_size, thread_limit)
            from .cute.thread_budget import check_thread_limit

            # Detect MMA-compatible K-loops: device loops containing
            # addmm/mm with float16/bfloat16 operands.
            mma_mode = False
            is_device_loop = any(bid not in grid_ids for bid in block_ids)
            if is_device_loop:
                mma_mode = _detect_specialized_mma_loop(
                    fn,
                    block_ids,
                    block_sizes=nd_block_size,
                    config=config,
                )

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
        block_size = functools.reduce(operator.mul, nd_block_size)
        # Resolve per-axis thread counts then flatten to a single total
        flat_num_threads = functools.reduce(
            operator.mul,
            (
                nt if nt > 0 else (int(bs) if isinstance(bs, int) else 0)
                for nt, bs in zip(num_threads_config, nd_block_size, strict=True)
            ),
            1,
        )
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

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()


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

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"tgid[{dim}]"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + tid[{axis}]"

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
