from __future__ import annotations

import abc
import dataclasses
import functools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

import torch

from .. import exc
from .ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    import sympy
    from torch._inductor.ops_handler import OpsHandler

    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .device_function import Argument
    from .device_function import DeviceFunction
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
        return f"tl.cast({expr_str}, {dtype_str})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        """Render a SymPy expression for this backend's device code."""
        from .device_function import texpr

        return texpr(expr)

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
        return f"{offsets_var} = {lid} * {block_size_var} + tl.arange(0, {block_size_var}).to({dtype})"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific grid index expression from an offset."""
        return f"({offset_var} + tl.arange(0, ({block_size_var}))).to({dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Generate backend-specific device-loop index expression from an offset."""
        return f"{offset_var} + tl.arange(0, ({block_size_var})).to({dtype})"

    def scalar_load_expr(self, tensor_name: str) -> str:
        """Load scalar value from a tensor argument."""
        return f"tl.load({tensor_name})"

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

    def grid_barrier_stmt(self, sem_arg: str) -> str:
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
        self, block_specs: list[object], ndim: int
    ) -> None:
        """Adjust block-size min/max constraints for backend-specific alignment.

        Called after all block-size specs have been created.  ``block_specs``
        is a list of ``BlockSizeSpec`` objects (one per tiled dimension).
        ``ndim`` is the total number of tiled dimensions.

        The default does nothing.  Backends with alignment requirements
        (e.g., Pallas/TPU) override this to enforce minimums.
        """
        return

    def pin_num_warps(self, ndim: int) -> int | None:
        """Return a fixed num_warps value to pin, or None to keep it tunable.

        Called during config spec finalization.  ``ndim`` is the number of
        tiled dimensions (block_sizes entries).  Backends where num_warps
        is a no-op for certain kernel shapes can return a fixed value to
        eliminate wasted autotuner search budget.
        """
        return None

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
        return f"tl.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        """Generate a backend-specific minimum expression."""
        return f"tl.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        """Generate a backend-specific arange expression for reduction index setup."""
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        """Generate a backend-specific zeros expression."""
        return f"tl.zeros({shape}, {dtype})"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        raise exc.BackendUnsupported(self.name, "full tensor creation")

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"tl.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"tl.broadcast_to({expr}, {shape})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        """Generate the index expression for a reduction dimension.

        For Triton this is tl.arange; for CuTe it maps to a thread index.
        """
        return f"tl.arange(0, {block_size_var}).to({dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        """Generate the zero-length index expression for an empty reduction."""
        return f"tl.zeros([0], {dtype})"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        """Generate a host-side next-power-of-2 expression."""
        return f"triton.next_power_of_2({expr})"

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
        device_fn: DeviceFunction | None = None,
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
                fragments["matrix_instr_nonkdim"] = EnumFragment(choices=(0, 16, 32))

        if supports_mtia_tunables():
            fragments.update(get_mtia_tunable_fragments())

        return fragments

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.utils import triton_type

        return triton_type(dtype)

    def acc_type(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.triton import triton_acc_type

        return triton_acc_type(dtype)

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
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "triton": "import triton",
            "tl": "import triton.language as tl",
            "triton_helpers": "from torch._inductor.runtime import triton_helpers",
            "tl_math": "from torch._inductor.runtime.triton_helpers import math as tl_math",
            "libdevice": "from torch._inductor.runtime.triton_compat import libdevice",
            "_default_launcher": "from helion.runtime import default_launcher as _default_launcher",
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
        device_fn: DeviceFunction | None = None,
    ) -> list[str]:
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out


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
    "torch.bool": "jnp.bool_",
    "torch.complex64": "jnp.complex64",
    "torch.complex128": "jnp.complex128",
}


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend for TPU."""

    @property
    def name(self) -> str:
        return "pallas"

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
            "reduction_loops",
            "pallas_loop_type",
        }
    )

    def supports_config_key(self, key: str) -> bool:
        return key in self._PALLAS_SUPPORTED_KEYS

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"pl.program_id({dim})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"lax.convert_element_type({expr_str}, {dtype_str})"

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
            device_expr = (
                f"{tensor_host_args[0]}.device" if tensor_host_args else "'tpu'"
            )
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

    def scalar_load_expr(self, tensor_name: str) -> str:
        return f"{tensor_name}[0]"

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

    def adjust_block_size_constraints(
        self, block_specs: list[object], ndim: int
    ) -> None:
        """Enforce TPU alignment on block sizes.

        TPU Pallas requires:
        - Last dim block size: multiple of 128
        - Second-to-last dim block size: multiple of 8
        """
        from ..autotuner.config_spec import BlockSizeSpec

        for i, spec in enumerate(block_specs):
            if not isinstance(spec, BlockSizeSpec):
                continue
            dim_from_end = ndim - 1 - i
            if dim_from_end == 0:
                # Last dimension: must be multiple of 128
                spec.update_min(128)
            elif dim_from_end == 1:
                # Second-to-last dimension: must be multiple of 8
                spec.update_min(8)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from ..autotuner.config_fragment import EnumFragment
        from ..autotuner.config_spec import VALID_PALLAS_LOOP_TYPES

        return {"pallas_loop_type": EnumFragment(choices=VALID_PALLAS_LOOP_TYPES)}

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
        # inductor_prims.seeds uses torch.randint with int64 which is not
        # supported on XLA/TPU.  Generate on CPU then cast to int32 (required
        # by Mosaic lowering) and move to the accelerator device.
        return f"inductor_prims.seeds({count}, torch.device('cpu')).to(torch.int32).to(torch.accelerator.current_accelerator())"

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
        device_fn: DeviceFunction | None = None,
    ) -> list[str]:
        # Determine which arg positions are outputs.  A tensor is an output if:
        #   1. It was created inside the function body (not in input_sources), OR
        #   2. It is a function parameter that is mutated in-place (e.g. x[tile] += ...)
        from .ast_read_writes import ReadWrites
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg
        from .host_function import HostFunction

        output_indices: list[int] = []
        if sorted_args is not None:
            env = CompileEnvironment.current()
            host_fn = HostFunction.current()
            mutated_params = set(ReadWrites.from_list(host_fn.body).writes) & {
                a.arg for a in host_fn.args.args
            }
            for i, arg in enumerate(sorted_args):
                if not isinstance(arg, TensorArg):
                    continue
                if arg.fake_value not in env.input_sources:
                    # Tensor created inside the function body (output)
                    output_indices.append(i)
                elif arg.host_str() in mutated_params:
                    # Input tensor mutated in-place
                    output_indices.append(i)

        launcher_args = [*args, f"_output_indices={output_indices}"]

        if has_rng_ops:
            launcher_args.insert(-1, "_rng_seed_buffer")

        # Pass scratch shapes for pipeline/fori_loop launcher
        pallas_loop_type = config.get("pallas_loop_type", "default")
        if pallas_loop_type in ("emit_pipeline", "fori_loop"):
            from .device_function import DeviceFunction

            device_fn = DeviceFunction.current()
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


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

    def supports_config_key(self, key: str) -> bool:
        if key == "elements_per_thread":
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
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        return CuteDSLOpOverrides()

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{dtype_str}({expr_str})"

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

    def scalar_load_expr(self, tensor_name: str) -> str:
        return f"{tensor_name}[0]"

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
        max_axis = max(axis_sizes)
        stride = 1
        terms: list[str] = []
        for axis in range(max_axis + 1):
            term = self.cast_expr(f"cute.arch.thread_idx()[{axis}]", index_type)
            if stride != 1:
                term = f"({term}) * {self.cast_expr(repr(stride), index_type)}"
            terms.append(term)
            stride *= axis_sizes.get(axis, 1)
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
        from .device_function import DeviceFunction

        codegen = DeviceFunction.current().codegen
        dims = tuple(codegen.max_thread_block_dims)
        if dims == (1, 1, 1):
            dim_exprs = DeviceFunction.current().tile_strategy.thread_block_dim_exprs()
            if dim_exprs is not None and dim_exprs != ("1", "1", "1"):
                return [f"block=({dim_exprs[0]}, {dim_exprs[1]}, {dim_exprs[2]})"]
            dims = DeviceFunction.current().tile_strategy.thread_block_dims()
        if dims[0] * dims[1] * dims[2] > 1024:
            raise exc.BackendUnsupported(
                self.name,
                f"thread block too large for cute kernel: {tuple(dims)}",
            )
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
        device_fn: DeviceFunction | None = None,
    ) -> list[str]:
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        if not tensor_host_args:
            raise exc.BackendUnsupported(self.name, "kernel launch without tensor args")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

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
        elements_per_thread = [
            int(
                env.config_spec.elements_per_thread.config_get(
                    config.elements_per_thread, block_id, 1
                )
            )
            for block_id in block_ids
        ]
        if (
            has_device_loops
            or has_dynamic_shape
            or len(device_ir.grid_block_ids) != 1
            or (len(block_ids) > 1 and not flattened)
        ):
            nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
            int_positions = [
                i for i, bs in enumerate(nd_block_size) if isinstance(bs, int)
            ]
            static_threads = functools.reduce(
                operator.mul,
                (
                    int(nd_block_size[i]) // elements_per_thread[i]
                    for i in int_positions
                ),
                1,
            )
            if static_threads > 1024:
                raise exc.BackendUnsupported(
                    self.name,
                    f"thread block too large for cute kernel: {tuple(nd_block_size)}",
                )
            return CuteNDTileStrategy(
                fn,
                block_ids,
                block_size=nd_block_size,
                loop_order=loop_order,
                l2_grouping=l2_grouping,
                elements_per_thread=elements_per_thread,
            )
        flat_elements_per_thread = functools.reduce(
            operator.mul, elements_per_thread, 1
        )
        block_size = functools.reduce(
            operator.mul, [bs.from_config_assert(config) for bs in block_size_infos]
        )
        if isinstance(block_size, int):
            physical_threads = block_size // max(flat_elements_per_thread, 1)
            if physical_threads > 1024:
                raise exc.BackendUnsupported(
                    self.name,
                    f"thread block too large for cute kernel: {block_size}",
                )
        return CuteFlattenedTileStrategy(
            fn,
            block_ids,
            block_size=block_size,
            loop_order=loop_order,
            elements_per_thread=flat_elements_per_thread,
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


# Metal Shading Language type mappings
DTYPE_TO_METAL: dict[torch.dtype, str] = {
    torch.float16: "half",
    torch.bfloat16: "bfloat",
    torch.float32: "float",
    torch.float64: "double",  # limited support on Metal
    torch.int8: "char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint8: "uchar",
    torch.bool: "bool",
}

# Accumulator type mapping for reductions (promote to float32 for stability)
METAL_ACC_TYPE: dict[torch.dtype, str] = {
    torch.float16: "float",
    torch.bfloat16: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.int8: "int",
    torch.int16: "int",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint8: "uint",
    torch.bool: "int",
}


# Apple GPU SIMD group width (threads per simdgroup)
METAL_SIMD_WIDTH: int = 32

# Only block_sizes and num_warps have effect in the Metal emitter.
# loop_orders, flatten_loops, reduction_loops are NOT used — excluding them
# prevents the autotuner from wasting search budget on no-op dimensions.
_METAL_SUPPORTED_KEYS: frozenset[str] = frozenset(
    {
        "block_sizes",
        "num_warps",
        "use_tg_cache",
    }
)


# Mapping from Triton tl.* dtype attribute names to Metal type strings
_TL_DTYPE_TO_METAL: dict[str, str] = {
    "float16": "half",
    "bfloat16": "bfloat",
    "float32": "float",
    "float64": "double",
    "int8": "char",
    "int16": "short",
    "int32": "int",
    "int64": "long",
    "uint8": "uchar",
}


class MetalKernelKind:
    """Classification of a Metal kernel for dispatch and codegen."""

    ELEMENTWISE = "elementwise"  # 1D flat dispatch
    SOFTMAX = "softmax"  # 2D row-parallel (one threadgroup per row)
    MATMUL = "matmul"  # 2D MPP matmul2d
    FUSED_ATTENTION = "fused_attention"  # matmul + softmax + matmul


@dataclasses.dataclass
class MetalMatmulOp:
    """Structured IR node for a matmul operation in the Metal backend.

    Names refer to tensor arguments or load variables. The actual
    shapes are resolved from ``TensorArg.fake_value`` at emit time
    (not stored here, since FX node shapes are per-tile, not full).
    """

    lhs_name: str
    rhs_name: str
    has_acc: bool
    acc_name: str | None
    is_batched: bool
    dtype: torch.dtype


@dataclasses.dataclass
class MetalReductionOp:
    """Structured IR node for a reduction operation in the Metal backend."""

    input_name: str
    reduction_type: str  # "sum", "max", "min"
    dim: int


class MslWalker:
    """AST-walking MSL emitter using MPP cooperative_tensor for all ops.

    Walks the body AST statement-by-statement, emitting MSL from composable
    building blocks.  Uses MPP ``matmul2d`` with ``cooperative_tensor`` and
    ``execution_simdgroup`` scope (1 SG per tile, enables ``reduce_rows``).

    For matmul-only kernels (no reductions), uses the multi-SG
    ``execution_simdgroups<N>`` path for higher throughput.

    For matmul + reduction (attention), uses ``execution_simdgroup`` with
    ``sg_idx`` indexing for multiple tiles per threadgroup.

    Statement → MSL mapping:
      - matmul sentinel → matmul2d.run() into cooperative_tensor
      - _metal_max/sum/min → reduce_rows() on cooperative_tensor
      - element-wise on coop var → begin()/end() iteration
      - map_iterator for row broadcast (x - m[:, None])
      - subscript store → coop.store() to device tensor
      - for loop → MSL for loop, recurse into body
      - scalar assignment → MSL scalar declaration
    """

    def __init__(
        self,
        backend: MetalBackend,
        device_fn: DeviceFunction,
        env: object,
        metal_ops: list[MetalMatmulOp | MetalReductionOp],
    ) -> None:
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        assert isinstance(env, CompileEnvironment)
        self.backend = backend
        self.device_fn = device_fn
        self.env = env
        self.metal_ops = metal_ops

        # Build arg maps
        self.tensor_args = [
            a for a in device_fn.sorted_args() if isinstance(a, TensorArg)
        ]
        self.arg_map: dict[str, TensorArg] = {a.name: a for a in self.tensor_args}
        self.load_to_arg: dict[str, str]
        self.out_buf: str | None
        self.load_to_arg, self.out_buf = MetalBackend._build_arg_maps(
            device_fn.preamble + device_fn.body, self.arg_map
        )

    def _resolve_arg(self, name: str) -> object | None:
        return MetalBackend._resolve_name(name, self.arg_map, self.load_to_arg)

    def generate(self) -> str:
        """Generate complete MSL source for the kernel."""
        kind = self.backend._kernel_kind

        if kind in (MetalKernelKind.MATMUL, MetalKernelKind.FUSED_ATTENTION):
            return self._generate_walked()
        if kind == MetalKernelKind.SOFTMAX:
            return self._generate_reduction()
        # ELEMENTWISE
        return self._generate_elementwise()

    # ------------------------------------------------------------------
    # Unified walker (Phases 1-3)
    # ------------------------------------------------------------------

    def _generate_walked(self) -> str:
        """Generate MSL by walking the body AST with cooperative_tensor.

        Handles matmul-only, matmul+epilogue, and matmul+reduction+matmul
        (fused attention) through a single code path.
        """
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        assert isinstance(self.env, CompileEnvironment)
        env = self.env
        device_fn = self.device_fn

        mm_ops = [op for op in self.metal_ops if isinstance(op, MetalMatmulOp)]
        has_reduction = any(isinstance(op, MetalReductionOp) for op in self.metal_ops)
        is_attention = len(mm_ops) >= 2 and has_reduction

        first_mm = mm_ops[0]

        # Resolve dimensions from structured IR via shared helper
        M, N, K, is_batched, batch_val, lhs_arg, rhs_arg = (
            MetalBackend._resolve_matmul_dims(first_mm, self.arg_map, self.load_to_arg)
        )
        assert isinstance(lhs_arg, TensorArg)
        assert isinstance(rhs_arg, TensorArg)

        # Determine whether RHS is transposed (walker-specific, not needed by launcher)
        rhs_d0 = env.size_hint(rhs_arg.fake_value.size(-2))
        transpose_rhs = rhs_d0 != K

        # For attention, resolve the second matmul's RHS (V) and output
        last_rhs_arg = None
        out_arg = None
        if is_attention and len(mm_ops) >= 2:
            last_rhs_arg_obj = self._resolve_arg(mm_ops[-1].rhs_name)
            if isinstance(last_rhs_arg_obj, TensorArg):
                last_rhs_arg = last_rhs_arg_obj
            out_arg_obj = self.arg_map.get(self.out_buf or "")
            if isinstance(out_arg_obj, TensorArg):
                out_arg = out_arg_obj

        # Fallbacks for output arg
        if out_arg is None:
            for a in self.tensor_args:
                if a is not lhs_arg and a is not rhs_arg and a is not last_rhs_arg:
                    out_arg = a
                    break
        assert out_arg is not None

        # For matmul-only, also resolve output
        if not is_attention:
            # Find output arg: any tensor not lhs or rhs
            out_arg_mm = None
            for a in self.tensor_args:
                if a is not lhs_arg and a is not rhs_arg:
                    out_arg_mm = a
                    break
            if out_arg_mm is not None:
                out_arg = out_arg_mm

        # Output D for attention (last dimension of output)
        if is_batched:
            out_d1 = env.size_hint(out_arg.fake_value.size(-1))
        else:
            out_d1 = env.size_hint(out_arg.fake_value.size(1))

        metal_dtype = DTYPE_TO_METAL.get(lhs_arg.fake_value.dtype, "float")
        config = device_fn.config

        if is_attention:
            # Attention: tile_m is the M-tile, may be at index 1 for batched
            tile_m_idx = 1 if is_batched and len(config.block_sizes) > 1 else 0
            TILE_M = (
                config.block_sizes[tile_m_idx]
                if len(config.block_sizes) > tile_m_idx
                else 64
            )
            # TILE_N is the inner loop tile size (next block_size after tile_m)
            # Clamp to N: per-tile softmax requires full N visibility
            tile_n_idx = tile_m_idx + 1
            TILE_N = (
                min(config.block_sizes[tile_n_idx], N)
                if len(config.block_sizes) > tile_n_idx
                else N
            )
            # For correctness, force TILE_N = N (tiled softmax not yet supported)
            TILE_N = N
            NUM_SG = config.num_warps if config.num_warps is not None else 4
        else:
            TILE_M = config.block_sizes[0] if len(config.block_sizes) > 0 else 64
            TILE_N = config.block_sizes[1] if len(config.block_sizes) > 1 else 32
            TILE_K = config.block_sizes[2] if len(config.block_sizes) > 2 else K
            TILE_K = min(TILE_K, K)  # clamp to actual K
            needs_k_loop = TILE_K < K
            NUM_SG = config.num_warps if config.num_warps is not None else 4

        # --- Emit MSL header ---
        msl: list[str] = [
            "#include <metal_stdlib>",
            "#include <metal_tensor>",
        ]
        if is_attention:
            msl.append(
                "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>"
            )
        else:
            msl.append("#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>")
        msl.extend(
            [
                "using namespace metal;",
                "using namespace mpp::tensor_ops;",
                "",
            ]
        )

        # --- Constants ---
        msl.extend(
            [
                f"constant int _M = {M};",
                f"constant int _N = {N};",
                f"constant int _K = {K};",
                f"constant int _TILE_M = {TILE_M};",
            ]
        )
        if is_attention:
            msl.extend(
                [
                    f"constant int _TILE_N = {TILE_N};",
                    f"constant int _OUT_D = {out_d1};",
                ]
            )
        else:
            msl.extend(
                [
                    f"constant int _TILE_N = {TILE_N};",
                    f"constant int _TILE_K = {TILE_K};",
                    f"constant int _NUM_SG = {NUM_SG};",
                ]
            )
        if is_batched:
            msl.append(f"constant int _BATCH = {batch_val};")
        msl.append("")

        # --- Kernel signature ---
        params: list[str] = []
        buf_idx = 0
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {dt}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1

        if is_attention:
            # Scratch buffer for materializing P between matmuls
            params.append(f"device {metal_dtype}* _scratch [[buffer({buf_idx})]]")
            tgid_type = "uint3" if is_batched else "uint2"
            params.extend(
                [
                    f"{tgid_type} tgid [[threadgroup_position_in_grid]]",
                    "uint _sg_idx [[simdgroup_index_in_threadgroup]]",
                ]
            )
        else:
            params.append("uint2 tgid [[threadgroup_position_in_grid]]")

        sig = ", ".join(params)
        msl.extend([f"kernel void {device_fn.name}({sig}) {{", ""])

        if is_attention:
            # --- Attention path: execution_simdgroup, sg_idx tile indexing ---
            num_tiles = (M + TILE_M - 1) // TILE_M
            msl.extend(
                [
                    f"    uint _tile_row = tgid.y * {NUM_SG} + _sg_idx;",
                    f"    if (_tile_row >= {num_tiles}u) return;",
                    "",
                ]
            )

            # Per-head offsets for batched
            if is_batched:
                msl.extend(
                    [
                        "    // Per-head pointer offsets",
                        "    uint _head = tgid.z;",
                    ]
                )
                for arg in self.tensor_args:
                    dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                    stride = 1
                    for d in range(arg.fake_value.ndim - 1, 0, -1):
                        stride *= env.size_hint(arg.fake_value.size(d))
                    msl.append(
                        f"    device {dt}* _h_{arg.name} = {arg.name} + _head * {stride};"
                    )
                scratch_stride = M * N
                msl.extend(
                    [
                        f"    device {metal_dtype}* _h_scratch = _scratch + _head * {scratch_stride};",
                        "",
                    ]
                )

            # Wrap buffers as tensor_inline 2D tensors
            msl.append("    // Wrap buffers as tensor_inline 2D tensors")
            for arg in self.tensor_args:
                dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                d0 = env.size_hint(arg.fake_value.size(-2))
                d1 = env.size_hint(arg.fake_value.size(-1))
                ptr = f"_h_{arg.name}" if is_batched else arg.name
                msl.append(
                    f"    auto _t_{arg.name} = tensor<device {dt}, "
                    f"dextents<int32_t, 2>, tensor_inline>("
                    f"\n        {ptr}, dextents<int32_t, 2>({d1}, {d0}));"
                )
            scratch_ptr = "_h_scratch" if is_batched else "_scratch"
            _scratch_decl = (
                f"    auto _t_scratch = tensor<device {metal_dtype}, "
                f"dextents<int32_t, 2>, tensor_inline>("
                f"\n        {scratch_ptr}, dextents<int32_t, 2>({N}, {M}));"
            )
            msl.extend([_scratch_decl, ""])

            transpose_str = "true" if transpose_rhs else "false"

            # Emit online softmax attention body
            self._emit_attention_body(
                msl,
                lhs_arg,
                rhs_arg,
                last_rhs_arg,
                out_arg,
                transpose_str,
                is_batched,
                TILE_M,
                TILE_N,
                N,
                K,
                out_d1,
            )
        else:
            # --- Matmul-only path: multi-SG, L2 swizzle ---
            # Scan body AST for epilogue ops on the matmul result
            stmts = device_fn.preamble + device_fn.body
            epilogue_ops = self._scan_for_epilogue(stmts)
            use_cooperative = len(epilogue_ops) > 0

            # Determine matmul mode: multiply_accumulate when we loop over K
            mm_mode = "multiply_accumulate" if needs_k_loop else "multiply"

            msl.extend(
                [
                    "    // Wrap raw buffers as tensor_inline 2D tensors",
                    f"    auto _A = tensor<device {metal_dtype}, dextents<int32_t, 2>, tensor_inline>(",
                    f"        {lhs_arg.name}, dextents<int32_t, 2>(_K, _M));",
                    f"    auto _B = tensor<device {metal_dtype}, dextents<int32_t, 2>, tensor_inline>(",
                    f"        {rhs_arg.name}, dextents<int32_t, 2>(_N, _K));",
                    f"    auto _C = tensor<device {metal_dtype}, dextents<int32_t, 2>, tensor_inline>(",
                    f"        {out_arg.name}, dextents<int32_t, 2>(_N, _M));",
                    "",
                    "    constexpr auto _desc = matmul2d_descriptor(",
                    "        _TILE_M, _TILE_N, _TILE_K,",
                    f"        false, false, false, matmul2d_descriptor::mode::{mm_mode});",
                    "    matmul2d<_desc, execution_simdgroups<_NUM_SG>> _op;",
                    "",
                    "    // Threadgroup swizzle for L2 cache locality",
                    "    constexpr uint _SW = 8;",
                    "    uint _gm = (_M + _TILE_M - 1) / _TILE_M;",
                    "    uint _gn = (_N + _TILE_N - 1) / _TILE_N;",
                    "    uint _ty = tgid.y;",
                    "    uint _tx = tgid.x;",
                    "    if (_gm >= _SW && (_gm % _SW) == 0) {",
                    "        uint _linear = tgid.y * _gn + tgid.x;",
                    "        uint _block = _linear / (_SW * _gn);",
                    "        uint _local = _linear - _block * (_SW * _gn);",
                    "        _tx = _local / _SW;",
                    "        _ty = _block * _SW + (_local % _SW);",
                    "    }",
                    "",
                ]
            )

            if use_cooperative:
                # Epilogue fusion path: matmul into cooperative_tensor
                msl.extend(
                    [
                        "    auto _As = _A.slice(0, _ty * _TILE_M);",
                        "    auto _Bs = _B.slice(_tx * _TILE_N, 0);",
                        "",
                        "    auto _coop = _op.get_destination_cooperative_tensor<",
                        "        decltype(_As), decltype(_Bs), float>();",
                    ]
                )
                if needs_k_loop:
                    # Zero-init coop, then accumulate over K tiles
                    msl.extend(
                        [
                            "    for (auto _it = _coop.begin(); _it != _coop.end(); _it++)",
                            "        *_it = 0.0f;",
                            "    for (int _tk = 0; _tk < _K; _tk += _TILE_K) {",
                            "        auto _Ak = _A.slice(_tk, _ty * _TILE_M);",
                            "        auto _Bk = _B.slice(_tx * _TILE_N, _tk);",
                            "        _op.run(_Ak, _Bk, _coop);",
                            "    }",
                        ]
                    )
                else:
                    msl.append("    _op.run(_As, _Bs, _coop);")
                msl.append("")
                for epi_desc, epi_msl in epilogue_ops:
                    msl.extend(
                        [
                            f"    // Epilogue: {epi_desc}",
                            "    for (auto _it = _coop.begin(); _it != _coop.end(); _it++)",
                            f"        *_it = {epi_msl};",
                            "",
                        ]
                    )
                msl.extend(
                    [
                        "    auto _Cs = _C.slice(_tx * _TILE_N, _ty * _TILE_M);",
                        "    _coop.store(_Cs);",
                    ]
                )
            else:
                # Direct output path: matmul writes to device memory
                msl.append("    auto _Cs = _C.slice(_tx * _TILE_N, _ty * _TILE_M);")
                if needs_k_loop:
                    # Zero-init output, then accumulate over K tiles
                    msl.extend(
                        [
                            "    // Zero-init output tile",
                            "    auto _coop = _op.get_destination_cooperative_tensor<",
                            "        decltype(_Cs), decltype(_Cs), float>();",
                            "    for (auto _it = _coop.begin(); _it != _coop.end(); _it++)",
                            "        *_it = 0.0f;",
                            "    _coop.store(_Cs);",
                            "    for (int _tk = 0; _tk < _K; _tk += _TILE_K) {",
                            "        auto _Ak = _A.slice(_tk, _ty * _TILE_M);",
                            "        auto _Bk = _B.slice(_tx * _TILE_N, _tk);",
                            "        _op.run(_Ak, _Bk, _Cs);",
                            "    }",
                        ]
                    )
                else:
                    msl.extend(
                        [
                            "    auto _As = _A.slice(0, _ty * _TILE_M);",
                            "    auto _Bs = _B.slice(_tx * _TILE_N, 0);",
                            "    _op.run(_As, _Bs, _Cs);",
                        ]
                    )

        msl.append("}")
        return "\n".join(msl)

    def _extract_attention_scale(
        self,
        stmts: list[ast.stmt],
    ) -> float:
        """Extract the qk_scale constant from the inner loop body.

        The DSL computes: qk_scale = (1/sqrt(d)) * 1.44269504 (log2(e))
        which appears as tl.full([], val, tl.float32) in the SSA form.
        """
        import ast as _ast

        for stmt in stmts:
            if not isinstance(stmt, _ast.For):
                continue
            for inner in stmt.body:
                if (
                    isinstance(inner, _ast.Assign)
                    and len(inner.targets) == 1
                    and isinstance(inner.targets[0], _ast.Name)
                ):
                    val = inner.value
                    if (
                        isinstance(val, _ast.Call)
                        and isinstance(val.func, _ast.Attribute)
                        and isinstance(val.func.value, _ast.Name)
                        and val.func.value.id == "tl"
                        and val.func.attr == "full"
                        and len(val.args) >= 2
                        and isinstance(val.args[0], _ast.List)
                        and len(val.args[0].elts) == 0
                        and isinstance(val.args[1], _ast.Constant)
                        and isinstance(val.args[1].value, float)
                    ):
                        return float(val.args[1].value)
        return 0.0

    def _emit_attention_body(
        self,
        msl: list[str],
        lhs_arg: object,
        rhs_arg: object,
        last_rhs_arg: object | None,
        out_arg: object,
        transpose_str: str,
        is_batched: bool,
        TILE_M: int,
        TILE_N: int,
        N: int,
        K: int,
        out_d1: int,
    ) -> None:
        """Emit tiled attention MSL with inner loop over N-dimension.

        Tiles the scores computation over TILE_N-sized chunks. Each tile
        computes partial scores, applies safe softmax (shift by tile max
        for numerical stability), and stores to the scratch P matrix at
        the correct offset. After all tiles, normalizes P by the global
        row sums, then computes output = P @ V.

        When TILE_N >= N (common case), the loop runs once and this
        degenerates to the original naive attention.
        """
        from .device_function import TensorArg

        assert isinstance(lhs_arg, TensorArg)
        assert isinstance(rhs_arg, TensorArg)
        assert isinstance(out_arg, TensorArg)

        lhs_name = lhs_arg.name
        rhs_name = rhs_arg.name
        out_name = out_arg.name
        last_rhs_name = (
            last_rhs_arg.name if isinstance(last_rhs_arg, TensorArg) else None
        )

        stmts = self.device_fn.preamble + self.device_fn.body
        scale = self._extract_attention_scale(stmts)
        scale_str = f"{scale}f" if scale != 0.0 else "rsqrt((float)_K)"

        # -- Declare Q slice and scores matmul --
        msl.extend(
            [
                f"    auto _q_slice = _t_{lhs_name}.slice(0, _tile_row * _TILE_M);",
                "",
                "    // Scores matmul descriptor (TILE_M x TILE_N tiles)",
                "    constexpr auto _scoreDesc = matmul2d_descriptor(",
                "        _TILE_M, _TILE_N, _K,",
                f"        false, {transpose_str}, false,",
                "        matmul2d_descriptor::mode::multiply);",
                "    matmul2d<_scoreDesc, execution_simdgroup> _scoreOp;",
                "",
            ]
        )

        # -- Inner loop: compute scores tiles, apply softmax, store to scratch --
        msl.extend(
            [
                "    for (int _tn = 0; _tn < _N; _tn += _TILE_N) {",
                "",
                "        // Scores = Q_tile @ K_tile",
                f"        auto _k_tile = _t_{rhs_name}.slice(_tn, 0);",
                "        auto _cScores = _scoreOp.get_destination_cooperative_tensor<",
                "            decltype(_q_slice), decltype(_k_tile), float>();",
                "        _scoreOp.run(_q_slice, _k_tile, _cScores);",
                "",
                "        // Scale scores",
                "        for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++)",
                f"            *_it *= {scale_str};",
                "",
                "        // Row max for numerical stability",
                "        auto _cMaxRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<",
                "            decltype(_q_slice), decltype(_k_tile), float>();",
                "        reduce_rows(_cScores, _cMaxRow, reduction_operation::max,",
                "            metal::numeric_limits<float>::lowest());",
                "",
                "        // Subtract max and exponentiate: exp2(score - max)",
                "        for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {",
                "            auto _max_it = _cMaxRow.map_iterator(_it);",
                "            *_it = exp2(*_it - *_max_it);",
                "        }",
                "",
                "        // Row sum of exponentials",
                "        auto _cSumRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<",
                "            decltype(_q_slice), decltype(_k_tile), float>();",
                "        reduce_rows(_cScores, _cSumRow, reduction_operation::sum, 0.0f);",
                "",
                "        // Normalize by row sum",
                "        for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {",
                "            auto _sum_it = _cSumRow.map_iterator(_it);",
                "            *_it *= (1.0f / *_sum_it);",
                "        }",
                "",
                "        // Store normalized tile to scratch at correct offset",
                "        auto _scratch_tile = _t_scratch.slice(_tn, _tile_row * _TILE_M);",
                "        _cScores.store(_scratch_tile);",
                "        simdgroup_barrier(mem_flags::mem_device);",
                "",
                "    }",
                "",
            ]
        )

        # -- Output matmul: P @ V → output --
        if last_rhs_name:
            msl.extend(
                [
                    "    // Output matmul: scratch(P) @ V -> output",
                    "    constexpr auto _outDesc = matmul2d_descriptor(",
                    "        _TILE_M, _OUT_D, dynamic_length_v<int>,",
                    "        false, false, false,",
                    "        matmul2d_descriptor::mode::multiply);",
                    "    matmul2d<_outDesc, execution_simdgroup> _outOp;",
                    "    auto _scratch_row = _t_scratch.slice(0, _tile_row * _TILE_M);",
                    f"    auto _v_slice = _t_{last_rhs_name}.slice(0, 0);",
                    f"    auto _o_slice = _t_{out_name}.slice(0, _tile_row * _TILE_M);",
                    "    _outOp.run(_scratch_row, _v_slice, _o_slice);",
                ]
            )
        else:
            msl.extend(
                [
                    f"    auto _o_slice = _t_{out_name}.slice(0, _tile_row * _TILE_M);",
                    "    auto _scratch_row = _t_scratch.slice(0, _tile_row * _TILE_M);",
                    "    // Store scratch directly to output (no second matmul)",
                    "    // (scratch data is already the final result)",
                ]
            )

    def _scan_for_epilogue(
        self,
        stmts: list[ast.stmt],
    ) -> list[tuple[str, str]]:
        """Scan AST for elementwise epilogue ops on the matmul result.

        Returns list of (description, msl_expression) pairs where
        msl_expression uses ``*_it`` for the cooperative_tensor element.

        The framework generates SSA-style names: the K-loop body assigns
        ``acc_1 = _metal_addmm(acc_copy_0, ...)`` but the epilogue after
        the loop references ``acc`` (the original accumulator variable).
        We track both the SSA result name and the original accumulator
        name as aliases for the matmul result.
        """
        import ast as _ast

        epilogue_ops: list[tuple[str, str]] = []

        # Find the matmul sentinel (may be inside a For loop body)
        # and the original accumulator variable it feeds into.
        all_stmts = self._flatten_stmts(stmts)

        # Find the matmul assignment and the accumulator name
        mm_result_var: str | None = None
        acc_var: str | None = None
        mm_idx = -1
        for i, stmt in enumerate(all_stmts):
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, _ast.Name) and self._is_matmul_sentinel(
                    stmt.value
                ):
                    mm_result_var = target.id
                    mm_idx = i
                    # The addmm sentinel has acc as first arg:
                    # _metal_addmm(acc_copy_0, load, load_1)
                    # The original accumulator is typically "acc" — extract
                    # from the acc_copy chain.
                    assert isinstance(stmt.value, _ast.Call)
                    if stmt.value.args:
                        acc_arg = stmt.value.args[0]
                        if isinstance(acc_arg, _ast.Name):
                            # Trace back: acc_copy_0 -> acc_copy -> acc
                            base = acc_arg.id
                            while base.endswith(("_0", "_copy")):
                                if base.endswith("_0"):
                                    base = base[:-2]
                                elif base.endswith("_copy"):
                                    base = base[:-5]
                            acc_var = base

        if mm_result_var is None or mm_idx < 0:
            return []

        # The matmul result can be referenced by mm_result_var or acc_var
        alias_vars = {mm_result_var}
        if acc_var:
            alias_vars.add(acc_var)

        # Walk stmts AFTER the for loop (not just after the matmul in
        # the flattened list). Use the original (non-flattened) stmts
        # to find post-loop operations.
        post_loop_stmts: list[_ast.stmt] = []
        found_for = False
        for stmt in stmts:
            if isinstance(stmt, _ast.For):
                found_for = True
                continue
            if found_for:
                post_loop_stmts.append(stmt)

        # If no For loop found, use stmts after the matmul in the flat list
        if not post_loop_stmts:
            post_loop_stmts = list(all_stmts[mm_idx + 1 :])

        # Walk post-loop statements for elementwise ops on the result.
        # Track local constants (e.g. v_0 = tl.full([], 0, ...)) so
        # we can resolve them in epilogue patterns like maximum(v_0, acc).
        current_vars = set(alias_vars)
        local_constants: dict[str, object] = {}
        for stmt in post_loop_stmts:
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, _ast.Name):
                    # Track tl.full([], val, dtype) as local constants
                    if self._is_scalar_full(stmt.value):
                        local_constants[target.id] = True
                        continue

                    # Try each alias
                    for cv in current_vars:
                        epi = self._extract_epilogue_op(stmt.value, cv, local_constants)
                        if epi is not None:
                            epilogue_ops.append(epi)
                            current_vars = {target.id}
                            break
                    else:
                        continue
                    continue
            if self._is_tile_store(stmt):
                break

        return epilogue_ops

    @staticmethod
    def _is_scalar_full(expr: ast.AST) -> bool:
        """Check if expression is tl.full([], val, dtype)."""
        import ast as _ast

        if not isinstance(expr, _ast.Call):
            return False
        func = expr.func
        if (
            isinstance(func, _ast.Attribute)
            and isinstance(func.value, _ast.Name)
            and func.value.id == "tl"
            and func.attr == "full"
            and len(expr.args) >= 1
        ):
            shape_arg = expr.args[0]
            return isinstance(shape_arg, _ast.List) and len(shape_arg.elts) == 0
        return False

    @staticmethod
    def _flatten_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Flatten For loop bodies into a single statement list."""
        import ast as _ast

        result: list[_ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, _ast.For):
                result.extend(MslWalker._flatten_stmts(stmt.body))
            else:
                result.append(stmt)
        return result

    def _is_matmul_sentinel(self, expr: ast.AST) -> bool:
        """Check if an expression is a matmul sentinel using structured IR.

        Matches by checking if the call's argument names correspond to any
        MetalMatmulOp's operand names (lhs_name, rhs_name).
        """
        import ast as _ast

        if not isinstance(expr, _ast.Call) or not isinstance(expr.func, _ast.Name):
            return False
        # Extract argument names from the call
        arg_names = [a.id for a in expr.args if isinstance(a, _ast.Name)]
        if len(arg_names) < 2:
            return False
        # Check against structured IR operand names
        for op in self.metal_ops:
            if isinstance(op, MetalMatmulOp):
                if op.lhs_name in arg_names and op.rhs_name in arg_names:
                    return True
        return False

    @staticmethod
    def _is_tile_store(stmt: ast.stmt) -> bool:
        """Check if a statement is a tile store (subscript assignment)."""
        import ast as _ast

        if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
            return isinstance(stmt.targets[0], _ast.Subscript)
        return False

    def _extract_epilogue_op(
        self,
        value: ast.AST,
        input_var: str,
        local_constants: dict[str, object] | None = None,
    ) -> tuple[str, str] | None:
        """Extract an elementwise epilogue op from an AST expression.

        Returns (description, msl_expr) where msl_expr uses ``*_it`` for
        the cooperative_tensor element, or None if not an epilogue op.
        """
        import ast as _ast

        if local_constants is None:
            local_constants = {}

        if isinstance(value, _ast.Call):
            func = value.func

            # relu: select(false_val, true_val, mask)
            if isinstance(func, _ast.Name) and func.id == "select":
                args = value.args
                if len(args) == 3 and self._expr_uses_var(args[1], input_var):
                    return ("relu", "max(*_it, 0.0f)")

            # relu: triton_helpers.maximum(const, x) or maximum(x, const)
            # Triton lowers torch.relu as maximum(tl.full([], 0), x)
            if isinstance(func, _ast.Attribute) and func.attr == "maximum":
                args = value.args
                if len(args) == 2:
                    if self._expr_uses_var(
                        args[1], input_var
                    ) and self._is_zero_or_const(args[0], local_constants):
                        return ("relu", "max(*_it, 0.0f)")
                    if self._expr_uses_var(
                        args[0], input_var
                    ) and self._is_zero_or_const(args[1], local_constants):
                        return ("relu", "max(*_it, 0.0f)")

            # tl.cast(x, dtype)
            if (
                isinstance(func, _ast.Attribute)
                and isinstance(func.value, _ast.Name)
                and func.value.id == "tl"
                and func.attr == "cast"
                and len(value.args) == 2
            ):
                if self._expr_uses_var(value.args[0], input_var):
                    dtype_node = value.args[1]
                    if isinstance(dtype_node, _ast.Attribute) and isinstance(
                        dtype_node.value, _ast.Name
                    ):
                        tl_dtype = dtype_node.attr
                        metal_type = _TL_DTYPE_TO_METAL.get(tl_dtype, "float")
                        return (
                            "cast",
                            f"static_cast<{metal_type}>(*_it)",
                        )

        # BinOp: x * scalar, x + scalar, etc.
        if isinstance(value, _ast.BinOp):
            if self._expr_uses_var(value.left, input_var) and not self._expr_uses_var(
                value.right, input_var
            ):
                rhs_msl = self.backend._ast_expr_to_msl(value.right)
                op_str = self._binop_to_str(value.op)
                if op_str:
                    return (
                        f"binop {op_str}",
                        f"(*_it {op_str} {rhs_msl})",
                    )
            if self._expr_uses_var(value.right, input_var) and not self._expr_uses_var(
                value.left, input_var
            ):
                lhs_msl = self.backend._ast_expr_to_msl(value.left)
                op_str = self._binop_to_str(value.op)
                if op_str:
                    return (
                        f"binop {op_str}",
                        f"({lhs_msl} {op_str} *_it)",
                    )

        return None

    @staticmethod
    def _is_zero_or_const(node: ast.AST, local_constants: dict[str, object]) -> bool:
        """Check if an AST node is a zero constant or a tracked local constant."""
        import ast as _ast

        if isinstance(node, _ast.Constant) and node.value in (0, 0.0):
            return True
        return isinstance(node, _ast.Name) and node.id in local_constants

    @staticmethod
    def _expr_uses_var(expr: ast.AST, var: str) -> bool:
        """Check if an expression references a variable by name."""
        import ast as _ast

        for node in _ast.walk(expr):
            if isinstance(node, _ast.Name) and node.id == var:
                return True
        return False

    @staticmethod
    def _binop_to_str(op: ast.AST) -> str | None:
        import ast as _ast

        if isinstance(op, _ast.Add):
            return "+"
        if isinstance(op, _ast.Sub):
            return "-"
        if isinstance(op, _ast.Mult):
            return "*"
        if isinstance(op, _ast.Div):
            return "/"
        return None

    # ------------------------------------------------------------------
    # Constexpr resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_constexpr_define(arg: object) -> str:
        """Resolve a ConstExprArg/NumericArgument to a proper MSL constant define.

        Handles two problems:
        1. Bare identifier host_str (e.g. "eps") would generate self-referential
           ``constant int eps = eps;`` — resolve to the actual numeric value.
        2. Always emitting ``int`` even for float parameters — detect floats and
           emit ``constant float`` instead.
        """
        import re

        from .compile_environment import CompileEnvironment
        from .device_function import Argument
        from .device_function import DeviceFunction
        from .device_function import SymbolArgument
        from .host_function import HostFunction

        assert isinstance(arg, Argument)
        name = arg.name
        host_str = arg.host_str()

        # Try to resolve bare identifiers to actual values
        value: object = None
        if re.match(r"^[A-Za-z_]\w*$", host_str):
            # host_str is a bare identifier — resolve from constexpr_args
            host_fn = HostFunction.current()
            if host_str in host_fn.constexpr_args:
                value = host_fn.constexpr_args[host_str]

            # Also try resolving SymbolArgument via var_hints in shape env
            if value is None and isinstance(arg, SymbolArgument):
                import sympy

                from .compile_environment import shape_env_var_hints

                env = CompileEnvironment.current()
                df = DeviceFunction.current()
                var_hints = shape_env_var_hints(env.shape_env)
                for sym, sym_arg in df._expr_args.items():
                    if sym_arg is arg and sym in var_hints:
                        raw = var_hints[sym]
                        if isinstance(raw, sympy.Float):
                            value = float(raw)
                        elif isinstance(raw, (sympy.Integer, sympy.Rational)):
                            value = int(raw)
                        elif isinstance(raw, (int, float)):
                            value = raw
                        break
        else:
            # host_str is already a literal or expression — try to parse it
            import contextlib

            with contextlib.suppress(ValueError, TypeError):
                value = int(host_str)
            if value is None:
                with contextlib.suppress(ValueError, TypeError):
                    value = float(host_str)

        if isinstance(value, float):
            # Format with enough precision, append 'f' suffix for MSL
            formatted = repr(value)
            if "e" in formatted or "E" in formatted:
                msl_val = f"{formatted}f"
            else:
                msl_val = f"{formatted}f" if "." in formatted else f"{value:.10g}f"
            return f"constant float {name} = {msl_val};"
        if isinstance(value, int):
            return f"constant int {name} = {value};"
        # Fallback: emit as-is (expression like _BLOCK_SIZE_1)
        return f"constant int {name} = {host_str};"

    # ------------------------------------------------------------------
    # Reduction generation (Phase 3) — delegates to existing pipeline
    # ------------------------------------------------------------------

    def _generate_reduction(self) -> str:
        """Generate reduction kernel via existing reduction pipeline."""
        from .compile_environment import CompileEnvironment
        from .device_function import ConstExprArg
        from .device_function import NumericArgument
        from .device_function import TensorArg

        env = self.env
        assert isinstance(env, CompileEnvironment)
        device_fn = self.device_fn

        msl_parts: list[str] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]

        constexpr_defines: list[str] = []
        params: list[str] = []
        buf_idx = 0
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                metal_dtype = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {metal_dtype}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
            elif isinstance(arg, (ConstExprArg, NumericArgument)):
                constexpr_defines.append(MslWalker._resolve_constexpr_define(arg))

        msl_parts.extend(constexpr_defines)

        # Infer _RDIM and _NROWS from the first 2D+ tensor
        rdim_val = 1
        nrows_val = 1
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg) and arg.fake_value.ndim >= 2:
                rdim_val = env.size_hint(arg.fake_value.size(-1))
                nrows_val = env.size_hint(arg.fake_value.size(0))
                break

        msl_parts.extend(
            [
                f"constant uint _RDIM = {rdim_val};",
                f"constant uint _NROWS = {nrows_val};",
            ]
        )

        params.extend(
            [
                "uint _tg_pos [[threadgroup_position_in_grid]]",
                "uint _tid [[thread_position_in_threadgroup]]",
                "uint _tg_size [[threads_per_threadgroup]]",
            ]
        )

        sig = ", ".join(params)
        msl_parts.append(f"kernel void {device_fn.name}({sig}) {{")

        # Classify body statements via AST analysis
        reduction_ops = [
            op for op in self.metal_ops if isinstance(op, MetalReductionOp)
        ]
        entries, buf_1d = self.backend._classify_reduction_stmts(
            device_fn.preamble + device_fn.body, reduction_ops
        )

        # Threadgroup memory row cache: cache the input row in threadgroup
        # memory to eliminate redundant device memory reads across reduction
        # passes.  Controlled by the autotuner via use_tg_cache.
        # Guard: RDIM must fit in 32KB threadgroup memory.
        import re as _re

        config = device_fn.config
        use_tg_cache = config.config.get("use_tg_cache", False)
        _max_tg_bytes = 32768

        tg_row_cache: dict[str, str] = {}
        if use_tg_cache and rdim_val * 4 <= _max_tg_bytes:
            # Find the first 2D load buffer in the entries
            _load_pat = _re.compile(r"auto \w+ = (\w+)\[_row \* _RDIM \+ _j\];")
            for kind, data, _ in entries:
                if kind == "col" and isinstance(data, str):
                    m = _load_pat.search(data.replace("_gid * _RDIM", "_row * _RDIM"))
                    if m:
                        input_buf = m.group(1)
                        tg_row_cache[input_buf] = "_tg_row"
                        break

        MetalBackend._emit_reduction_body(
            msl_parts, entries, buf_1d, tg_row_cache=tg_row_cache
        )

        msl_parts.append("}")
        return "\n".join(msl_parts)

    # ------------------------------------------------------------------
    # Elementwise generation (Phase 4)
    # ------------------------------------------------------------------

    def _generate_elementwise(self) -> str:
        """Generate a simple 1D per-thread elementwise kernel."""
        import ast as _ast

        from .device_function import ConstExprArg
        from .device_function import NumericArgument
        from .device_function import TensorArg

        device_fn = self.device_fn
        msl_parts: list[str] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]

        params: list[str] = []
        buf_idx = 0
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                metal_dtype = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {metal_dtype}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
            elif isinstance(arg, (ConstExprArg, NumericArgument)):
                msl_parts.append(MslWalker._resolve_constexpr_define(arg))
        params.append("uint _gid [[thread_position_in_grid]]")

        sig = ", ".join(params)
        msl_parts.append(f"kernel void {device_fn.name}({sig}) {{")

        for stmt in device_fn.preamble + device_fn.body:
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
                if isinstance(target, _ast.Name):
                    val_msl = self.backend._ast_expr_to_msl(value)
                    msl_parts.append(f"    auto {target.id} = {val_msl};")
                elif isinstance(target, _ast.Subscript):
                    target_msl = self.backend._ast_expr_to_msl(target)
                    val_msl = self.backend._ast_expr_to_msl(value)
                    msl_parts.append(f"    {target_msl} = {val_msl};")
                else:
                    msl_parts.append(
                        f"    {self.backend._ast_expr_to_msl(target)} = "
                        f"{self.backend._ast_expr_to_msl(value)};"
                    )
            elif isinstance(stmt, _ast.Expr):
                msl_parts.append(f"    {self.backend._ast_expr_to_msl(stmt.value)};")

        msl_parts.append("}")
        return "\n".join(msl_parts)


class MetalBackend(Backend):
    """Metal Shading Language (MSL) code generation backend for macOS MPS devices.

    Generates MSL C++ source code. The device function body produces MSL
    expression fragments. ``codegen_function_def`` assembles them into a
    complete MSL kernel and wraps the result in a Python function that
    returns ``(msl_source, kernel_name)``. The Metal launcher compiles
    the MSL via ``torch.mps.compile_shader()`` and dispatches.

    Metal mapping:
      - Tile / program instance → Metal threadgroup
      - program_id → ``threadgroup_position_in_grid``
      - Per-thread index → ``thread_position_in_grid`` (global)
      - Shared memory → ``threadgroup`` address space
    """

    def __init__(self) -> None:
        super().__init__()
        # Set by generate_msl_function, read by build_launcher_args
        self._kernel_kind: str = MetalKernelKind.ELEMENTWISE
        # Structured IR: populated during lowering, consumed by generate_msl_function
        self._metal_ops: list[MetalMatmulOp | MetalReductionOp] = []

    @property
    def name(self) -> str:
        return "metal"

    def dtype_str(self, dtype: torch.dtype) -> str:
        if dtype not in DTYPE_TO_METAL:
            raise exc.BackendUnsupported(self.name, f"dtype: {dtype}")
        return DTYPE_TO_METAL[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype not in METAL_ACC_TYPE:
            raise exc.BackendUnsupported(self.name, f"acc_type for: {dtype}")
        return METAL_ACC_TYPE[dtype]

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        return "uint"

    @property
    def function_decorator(self) -> str:
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

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
                "from helion.runtime import default_metal_launcher as _default_metal_launcher"
            ),
        }

    # --- expression generators (produce MSL C++ fragments) ---

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        if dim == 0:
            return "_tgpos"
        if dim == 1:
            return "_tgpos1"
        raise exc.BackendUnsupported(self.name, f"multi-dimensional grids (dim={dim})")

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        return f"(({numel} + {block_size} - 1) // {block_size})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"static_cast<{dtype_str}>({expr_str})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .device_function import texpr

        return texpr(expr)

    def arange_expr(
        self,
        begin: str,
        end: str,
        dtype: str,
        shape: Sequence[str] | None = None,
    ) -> str:
        # Per-thread: global thread id is the index
        return f"({begin} + _tid_in_tg)"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        # Use global thread ID directly — torch.mps.compile_shader auto-dispatches
        # threads based on the first tensor's numel, so _gid covers all elements.
        return "_gid"

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"select({false_val}, {true_val}, {mask})"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"(1 << (int({expr}) - 1).bit_length())"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.triton import TritonOverrides

        return TritonOverrides()

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        metal_type = DTYPE_TO_METAL[dtype]
        return f"(({metal_type})({value_expr}))"

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
            self._metal_ops.append(
                MetalReductionOp(
                    input_name=input_name,
                    reduction_type=reduction_type,
                    dim=dim,
                )
            )
            return f"_metal_{reduction_type}({input_name}, {dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    # --- AST-to-MSL pipeline ---

    def _ast_expr_to_msl(self, node: ast.AST, *, reduction_ctx: bool = False) -> str:
        """Recursively convert an AST expression node to MSL C++ string."""
        import ast as _ast

        if isinstance(node, _ast.Name):
            return node.id

        if isinstance(node, _ast.Constant):
            if isinstance(node.value, float):
                return repr(node.value)
            return str(node.value)

        if isinstance(node, _ast.UnaryOp):
            operand = self._ast_expr_to_msl(node.operand, reduction_ctx=reduction_ctx)
            if isinstance(node.op, _ast.USub):
                return f"(-{operand})"
            if isinstance(node.op, _ast.Not):
                return f"(!{operand})"
            if isinstance(node.op, _ast.Invert):
                return f"(~{operand})"
            return operand

        if isinstance(node, _ast.BinOp):
            left = self._ast_expr_to_msl(node.left, reduction_ctx=reduction_ctx)
            right = self._ast_expr_to_msl(node.right, reduction_ctx=reduction_ctx)
            if isinstance(node.op, _ast.FloorDiv):
                return f"({left} / {right})"
            if isinstance(node.op, _ast.Pow):
                # x ** 0.5 → sqrt(x)
                if isinstance(node.right, _ast.Constant) and node.right.value == 0.5:
                    return f"sqrt({left})"
                return f"pow({left}, {right})"
            # Use isinstance checks because AST nodes may be wrapped
            # (helion._compiler.ast_extension.Wrapper) and type() won't
            # match the base ast.* classes in a dict lookup.
            op = node.op
            if isinstance(op, _ast.Add):
                op_str = "+"
            elif isinstance(op, _ast.Sub):
                op_str = "-"
            elif isinstance(op, _ast.Mult):
                op_str = "*"
            elif isinstance(op, _ast.Div):
                op_str = "/"
            elif isinstance(op, _ast.Mod):
                op_str = "%"
            elif isinstance(op, _ast.BitAnd):
                op_str = "&"
            elif isinstance(op, _ast.BitOr):
                op_str = "|"
            elif isinstance(op, _ast.BitXor):
                op_str = "^"
            elif isinstance(op, _ast.LShift):
                op_str = "<<"
            elif isinstance(op, _ast.RShift):
                op_str = ">>"
            else:
                op_str = "+"
            return f"({left} {op_str} {right})"

        if isinstance(node, _ast.BoolOp):
            op_str = " && " if isinstance(node.op, _ast.And) else " || "
            parts = [
                self._ast_expr_to_msl(v, reduction_ctx=reduction_ctx)
                for v in node.values
            ]
            return f"({op_str.join(parts)})"

        if isinstance(node, _ast.Compare):
            # Detect static_cast < dtype > expr  (parsed as Compare by Python)
            if (
                isinstance(node.left, _ast.Name)
                and node.left.id == "static_cast"
                and len(node.ops) == 2
                and isinstance(node.ops[0], _ast.Lt)
                and isinstance(node.ops[1], _ast.Gt)
                and isinstance(node.comparators[0], _ast.Name)
            ):
                metal_type = node.comparators[0].id
                inner = self._ast_expr_to_msl(
                    node.comparators[1], reduction_ctx=reduction_ctx
                )
                return f"static_cast<{metal_type}>({inner})"

            left = self._ast_expr_to_msl(node.left, reduction_ctx=reduction_ctx)
            parts = [left]
            for op, comp in zip(node.ops, node.comparators, strict=True):
                if isinstance(op, _ast.Eq):
                    parts.append("==")
                elif isinstance(op, _ast.NotEq):
                    parts.append("!=")
                elif isinstance(op, _ast.Lt):
                    parts.append("<")
                elif isinstance(op, _ast.LtE):
                    parts.append("<=")
                elif isinstance(op, _ast.Gt):
                    parts.append(">")
                elif isinstance(op, _ast.GtE):
                    parts.append(">=")
                else:
                    parts.append("==")
                parts.append(self._ast_expr_to_msl(comp, reduction_ctx=reduction_ctx))
            return f"({' '.join(parts)})"

        if isinstance(node, _ast.IfExp):
            test = self._ast_expr_to_msl(node.test, reduction_ctx=reduction_ctx)
            body = self._ast_expr_to_msl(node.body, reduction_ctx=reduction_ctx)
            orelse = self._ast_expr_to_msl(node.orelse, reduction_ctx=reduction_ctx)
            return f"({test} ? {body} : {orelse})"

        if isinstance(node, _ast.Call):
            return self._ast_call_to_msl(node, reduction_ctx=reduction_ctx)

        if isinstance(node, _ast.Subscript):
            return self._ast_subscript_to_msl(node, reduction_ctx=reduction_ctx)

        if isinstance(node, _ast.Attribute):
            value = self._ast_expr_to_msl(node.value, reduction_ctx=reduction_ctx)
            return f"{value}.{node.attr}"

        if isinstance(node, _ast.Tuple):
            parts = [
                self._ast_expr_to_msl(e, reduction_ctx=reduction_ctx) for e in node.elts
            ]
            return ", ".join(parts)

        # Fallback: unparse
        return _ast.unparse(node)

    def _ast_call_to_msl(self, node: ast.AST, *, reduction_ctx: bool) -> str:
        """Convert an AST Call node to MSL."""
        import ast as _ast

        assert isinstance(node, _ast.Call)

        func = node.func

        # libdevice.func(x) → func(x), with exp → exp2 fast path
        # tl_math.func(x) → func(x) (same mapping as libdevice)
        if (
            isinstance(func, _ast.Attribute)
            and isinstance(func.value, _ast.Name)
            and func.value.id in ("libdevice", "tl_math")
        ):
            args_msl = [
                self._ast_expr_to_msl(a, reduction_ctx=reduction_ctx) for a in node.args
            ]
            if func.attr == "exp" and len(args_msl) == 1:
                return f"exp2({args_msl[0]} * 1.4426950408889634f)"
            return f"{func.attr}({', '.join(args_msl)})"

        # tl.cast(x, tl.float32) → static_cast<float>(x)
        if (
            isinstance(func, _ast.Attribute)
            and isinstance(func.value, _ast.Name)
            and func.value.id == "tl"
            and func.attr == "cast"
            and len(node.args) == 2
        ):
            x_msl = self._ast_expr_to_msl(node.args[0], reduction_ctx=reduction_ctx)
            # Get dtype from tl.float32 style
            dtype_node = node.args[1]
            if isinstance(dtype_node, _ast.Attribute) and isinstance(
                dtype_node.value, _ast.Name
            ):
                tl_dtype = dtype_node.attr
                metal_type = _TL_DTYPE_TO_METAL.get(tl_dtype, "float")
            else:
                metal_type = "float"
            return f"static_cast<{metal_type}>({x_msl})"

        # tl.reshape(x, shape) → x  (strip reshape)
        if (
            isinstance(func, _ast.Attribute)
            and isinstance(func.value, _ast.Name)
            and func.value.id == "tl"
            and func.attr == "reshape"
            and len(node.args) >= 1
        ):
            return self._ast_expr_to_msl(node.args[0], reduction_ctx=reduction_ctx)

        # tl.full([], val, dtype) → ((metal_type)(val))
        if (
            isinstance(func, _ast.Attribute)
            and isinstance(func.value, _ast.Name)
            and func.value.id == "tl"
            and func.attr == "full"
            and len(node.args) >= 2
        ):
            # Check for empty shape: first arg is []
            shape_arg = node.args[0]
            if isinstance(shape_arg, _ast.List) and len(shape_arg.elts) == 0:
                val_msl = self._ast_expr_to_msl(
                    node.args[1], reduction_ctx=reduction_ctx
                )
                if len(node.args) >= 3:
                    dtype_node = node.args[2]
                    if isinstance(dtype_node, _ast.Attribute) and isinstance(
                        dtype_node.value, _ast.Name
                    ):
                        metal_type = _TL_DTYPE_TO_METAL.get(dtype_node.attr, "float")
                    else:
                        metal_type = "float"
                else:
                    metal_type = "float"
                return f"(({metal_type})({val_msl}))"

        # _metal_max/sum/min(input, dim) → sentinel (consumed by classifier)
        if isinstance(func, _ast.Name) and func.id.startswith("_metal_"):
            args_msl = [
                self._ast_expr_to_msl(a, reduction_ctx=reduction_ctx) for a in node.args
            ]
            return f"{func.id}({', '.join(args_msl)})"

        # select(f, t, m) → select(f, t, m)
        if isinstance(func, _ast.Name) and func.id == "select":
            args_msl = [
                self._ast_expr_to_msl(a, reduction_ctx=reduction_ctx) for a in node.args
            ]
            return f"select({', '.join(args_msl)})"

        # x.to(dtype) → strip, return x
        if isinstance(func, _ast.Attribute) and func.attr == "to":
            return self._ast_expr_to_msl(func.value, reduction_ctx=reduction_ctx)

        # Generic function call
        func_msl = self._ast_expr_to_msl(func, reduction_ctx=reduction_ctx)
        args_msl = [
            self._ast_expr_to_msl(a, reduction_ctx=reduction_ctx) for a in node.args
        ]
        return f"{func_msl}({', '.join(args_msl)})"

    def _ast_subscript_to_msl(self, node: ast.AST, *, reduction_ctx: bool) -> str:
        """Convert an AST Subscript node to MSL."""
        import ast as _ast

        assert isinstance(node, _ast.Subscript)

        buf_name = self._ast_expr_to_msl(node.value, reduction_ctx=reduction_ctx)
        sl = node.slice

        if isinstance(sl, _ast.Tuple):
            elts = sl.elts
            # x[idx, :] → x[_row * _RDIM + _j] (reduction ctx)
            # or x[_gid * _RDIM + _j] (elementwise ctx)
            if len(elts) >= 2 and isinstance(elts[-1], _ast.Slice):
                if all(isinstance(e, _ast.Slice) for e in elts):
                    # x[:, :] → 1D broadcast (detected by classifier)
                    return f"{buf_name}[:, :]"
                # x[idx, :] or x[batch, idx, :]
                if reduction_ctx:
                    return f"{buf_name}[_row * _RDIM + _j]"
                idx = self._ast_expr_to_msl(elts[-2], reduction_ctx=reduction_ctx)
                return f"{buf_name}[{idx} * _RDIM + _j]"
            # Regular tuple subscript: x[i, j]
            parts = [
                self._ast_expr_to_msl(e, reduction_ctx=reduction_ctx) for e in elts
            ]
            return f"{buf_name}[{', '.join(parts)}]"

        if isinstance(sl, _ast.Slice):
            return f"{buf_name}[:]"

        idx = self._ast_expr_to_msl(sl, reduction_ctx=reduction_ctx)
        if not reduction_ctx:
            return f"{buf_name}[{idx}]"
        return f"{buf_name}[{idx}]"

    def _classify_reduction_stmts(
        self,
        stmts: list[ast.stmt],
        reduction_ops: list[MetalReductionOp],
    ) -> tuple[list[tuple[str, str | tuple[str, str], str | None]], dict[str, str]]:
        """Classify body statements using AST analysis for _emit_reduction_body.

        Returns (classified_entries, buf_1d_map).
        Each entry is (kind, data, var_name):
          - ("col", msl_line, var_name|None)
          - ("scalar", msl_line, None)
          - ("reduce", (var_name, reduction_type), None)
          - ("store", msl_line, None)
        """
        import ast as _ast

        buf_1d: dict[str, str] = {}
        entries: list[tuple[str, str | tuple[str, str], str | None]] = []
        col_live: set[str] = set()
        red_idx = 0

        def _names_in_expr(node: _ast.AST) -> set[str]:
            """Collect all Name.id references in an AST expression."""
            names: set[str] = set()
            for child in _ast.walk(node):
                if isinstance(child, _ast.Name):
                    names.add(child.id)
            return names

        def _is_row_col_load(node: _ast.AST) -> bool:
            """Check if node is x[idx, :] pattern."""
            if not isinstance(node, _ast.Subscript):
                return False
            sl = node.slice
            if isinstance(sl, _ast.Tuple) and len(sl.elts) >= 2:
                return isinstance(sl.elts[-1], _ast.Slice) and not all(
                    isinstance(e, _ast.Slice) for e in sl.elts
                )
            return False

        def _is_1d_broadcast(node: _ast.AST) -> bool:
            """Check if node is x[:, :] pattern."""
            if not isinstance(node, _ast.Subscript):
                return False
            sl = node.slice
            if isinstance(sl, _ast.Tuple):
                return all(isinstance(e, _ast.Slice) for e in sl.elts)
            return False

        def _is_tl_full_scalar(node: _ast.AST) -> bool:
            """Check if node is tl.full([], val, dtype)."""
            if not isinstance(node, _ast.Call):
                return False
            func = node.func
            if (
                isinstance(func, _ast.Attribute)
                and isinstance(func.value, _ast.Name)
                and func.value.id == "tl"
                and func.attr == "full"
                and len(node.args) >= 1
            ):
                shape_arg = node.args[0]
                return isinstance(shape_arg, _ast.List) and len(shape_arg.elts) == 0
            return False

        # Build a set of reduction input names from structured IR
        # so we can identify reduction calls by their operands
        _red_input_names = {op.input_name for op in reduction_ops}

        def _find_reduction_op(node: _ast.AST) -> MetalReductionOp | None:
            """Find a reduction call by matching operands against structured IR.

            Walks the AST to handle wrapping (static_cast, tl.reshape).
            """
            for child in _ast.walk(node):
                if not isinstance(child, _ast.Call):
                    continue
                # Check if first arg references a known reduction input
                if child.args:
                    first_arg = child.args[0]
                    if (
                        isinstance(first_arg, _ast.Name)
                        and first_arg.id in _red_input_names
                    ):
                        # Match against specific reduction op
                        for op in reduction_ops:
                            if op.input_name == first_arg.id:
                                return op
            return None

        def _has_col_ref(node: _ast.AST) -> bool:
            """Check if expr references any column-live variable."""
            return bool(_names_in_expr(node) & col_live)

        for stmt in stmts:
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value

                # Emit indices_ as alias for _row (tile index = threadgroup position)
                if isinstance(target, _ast.Name) and target.id.startswith("indices_"):
                    entries.append(
                        ("scalar", f"    int {target.id} = (int)_row;", None)
                    )
                    continue

                # Reduction: match against structured IR operand names
                if isinstance(target, _ast.Name):
                    red_op = _find_reduction_op(value)
                    if red_op is not None:
                        var_name = target.id
                        red_type = red_op.reduction_type
                        col_live.discard(var_name)
                        entries.append(("reduce", (var_name, red_type), None))
                        red_idx += 1
                        continue

                # 1D broadcast: x[:, :]
                if isinstance(target, _ast.Name) and _is_1d_broadcast(value):
                    assert isinstance(value, _ast.Subscript)
                    buf_name = self._ast_expr_to_msl(value.value, reduction_ctx=True)
                    buf_1d[target.id] = buf_name
                    continue

                # Scalar init: tl.full([], val, dtype)
                if isinstance(target, _ast.Name) and _is_tl_full_scalar(value):
                    msl = self._ast_expr_to_msl(value, reduction_ctx=True)
                    entries.append(("scalar", f"    float {target.id} = {msl};", None))
                    continue

                # Store: out[idx, :] = val  (target is subscript)
                if isinstance(target, _ast.Subscript):
                    target_msl = self._ast_expr_to_msl(target, reduction_ctx=True)
                    val_msl = self._ast_expr_to_msl(value, reduction_ctx=True)
                    line = f"{target_msl} = {val_msl};"
                    # Stores referencing column-live vars are columnar
                    if _has_col_ref(value) or _is_row_col_load(target):
                        entries.append(("col", line, None))
                    else:
                        entries.append(("scalar", f"    {line}", None))
                    continue

                # Regular assignment
                if isinstance(target, _ast.Name):
                    var_name = target.id
                    val_msl = self._ast_expr_to_msl(value, reduction_ctx=True)

                    # Determine if column-live
                    is_col = _is_row_col_load(value) or _has_col_ref(value)
                    if is_col:
                        col_live.add(var_name)
                        line = f"auto {var_name} = {val_msl};"
                        entries.append(("col", line, var_name))
                    else:
                        line = f"auto {var_name} = {val_msl};"
                        entries.append(("scalar", f"    {line}", None))
                    continue

            # Expression statement (no assignment)
            if isinstance(stmt, _ast.Expr):
                msl = self._ast_expr_to_msl(stmt.value, reduction_ctx=True)
                entries.append(("scalar", f"    {msl};", None))
                continue

            # Fallback: walk AST expression
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
                if isinstance(target, _ast.Name):
                    val_msl = self._ast_expr_to_msl(value, reduction_ctx=True)
                    entries.append(
                        ("scalar", f"    auto {target.id} = {val_msl};", None)
                    )
                elif isinstance(target, _ast.Subscript):
                    target_msl = self._ast_expr_to_msl(target, reduction_ctx=True)
                    val_msl = self._ast_expr_to_msl(value, reduction_ctx=True)
                    entries.append(("scalar", f"    {target_msl} = {val_msl};", None))
                else:
                    msl = _ast.unparse(stmt).strip()
                    entries.append(("scalar", f"    {msl};", None))
            else:
                msl = _ast.unparse(stmt).strip()
                entries.append(("scalar", f"    {msl};", None))

        return entries, buf_1d

    @staticmethod
    def _build_arg_maps(
        stmts: list[ast.stmt],
        arg_map: dict[str, object],
    ) -> tuple[dict[str, str], str | None]:
        """Build load_to_arg map and find output buffer from AST.

        Walks the entire AST tree (including loop bodies) to find
        all load indirections and the output buffer.

        Returns (load_to_arg, out_buf_name).
        """
        import ast as _ast

        load_to_arg: dict[str, str] = {}
        out_buf: str | None = None

        for node in _ast.walk(_ast.Module(body=list(stmts), type_ignores=[])):
            if not isinstance(node, _ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value

            # Load indirection: load_name = buf_name[...]
            if (
                isinstance(target, _ast.Name)
                and isinstance(value, _ast.Subscript)
                and isinstance(value.value, _ast.Name)
            ):
                load_to_arg[target.id] = value.value.id

            # Output buffer: buf_name[...] = expr
            if (
                out_buf is None
                and isinstance(target, _ast.Subscript)
                and isinstance(target.value, _ast.Name)
                and target.value.id in arg_map
            ):
                out_buf = target.value.id

        return load_to_arg, out_buf

    def _classify_kernel(self) -> str:
        """Classify kernel type from _metal_ops. Returns MetalKernelKind constant."""
        has_matmul = any(isinstance(op, MetalMatmulOp) for op in self._metal_ops)
        has_reduction = any(isinstance(op, MetalReductionOp) for op in self._metal_ops)
        matmul_count = sum(1 for op in self._metal_ops if isinstance(op, MetalMatmulOp))
        if has_matmul and has_reduction and matmul_count >= 2:
            # Verify matmul→reduction→matmul ordering: the first reduction
            # must appear after a matmul and before the last matmul.
            saw_matmul = False
            saw_reduction_after_matmul = False
            saw_matmul_after_reduction = False
            for op in self._metal_ops:
                if isinstance(op, MetalMatmulOp):
                    if saw_reduction_after_matmul:
                        saw_matmul_after_reduction = True
                    saw_matmul = True
                elif isinstance(op, MetalReductionOp) and saw_matmul:
                    saw_reduction_after_matmul = True
            if saw_matmul_after_reduction:
                return MetalKernelKind.FUSED_ATTENTION
            # Ordering doesn't match attention pattern; fall through to MATMUL
        if has_matmul:
            return MetalKernelKind.MATMUL
        if has_reduction:
            return MetalKernelKind.SOFTMAX
        return MetalKernelKind.ELEMENTWISE

    # --- MSL assembly ---

    def generate_msl_function(self, device_fn: DeviceFunction) -> list[ast.stmt]:
        """Build a Python function that returns ``(msl_source, kernel_name)``.

        Called from ``DeviceFunction.codegen_function_def()`` when the
        backend is Metal.  Classifies the kernel from the structured
        ``_metal_ops`` IR, then dispatches to the appropriate generator.
        """
        import ast as _ast

        from .ast_extension import create
        from .ast_extension import create_arguments
        from .ast_extension import statement_from_string

        kernel_name = device_fn.name

        # Consume ops recorded during lowering
        metal_ops = list(self._metal_ops)
        self._metal_ops = []

        # Temporarily set _metal_ops so _classify_kernel can read them
        self._metal_ops = metal_ops
        kind = self._classify_kernel()
        self._metal_ops = []

        if kind == MetalKernelKind.FUSED_ATTENTION:
            matmul_count = sum(1 for op in metal_ops if isinstance(op, MetalMatmulOp))
            if matmul_count < 2:
                raise exc.BackendUnsupported(
                    "metal",
                    f"matmul+reduction kernel has {matmul_count} matmul(s), "
                    "but fused attention requires at least 2 (scores + output). "
                    "Single matmul + reduction is not yet supported.",
                )
        self._kernel_kind = kind

        # Store ops for build_launcher_args (called after generate_msl_function)
        self._last_metal_ops = metal_ops

        msl_source = self._generate_kernel(device_fn, metal_ops)

        fn_body = statement_from_string(f"return ({msl_source!r}, {kernel_name!r})")
        fn_def = create(
            _ast.FunctionDef,
            name=kernel_name,
            args=create_arguments([]),
            body=[fn_body],
            decorator_list=[],
            type_params=[],
        )
        return [fn_def]

    def _generate_kernel(
        self,
        device_fn: DeviceFunction,
        metal_ops: list[MetalMatmulOp | MetalReductionOp],
    ) -> str:
        """Dispatch to the appropriate kernel generator based on classification."""
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()

        # All kernel types route through the MslWalker.
        # FUSED_ATTENTION uses the walker's unified cooperative_tensor path.
        walker = MslWalker(self, device_fn, env, metal_ops)
        return walker.generate()

    @staticmethod
    def _optimize_reduction_entries(
        entries: list[tuple[str, str | tuple[str, str], str | None]],
        *,
        use_tg_cache: bool = False,
    ) -> list[tuple[str, str | tuple[str, str], str | None]]:
        """Optimize reduction entries to eliminate redundant recomputation.

        Handles two patterns:

        1. Single-reduction (softmax): col → reduce → col(recompute) → store.
           Stores intermediate before reduction, loads back after.
           Turns 3-pass into 2-pass.

        2. Two-reduction (LayerNorm) with use_tg_cache:
           col(load x, sum) → reduce1(mean) →
           col(center, square, sum) → reduce2(var) →
           col(re-center, normalize) → store
           Stores centered values to out_buf in pass 2, reads from out_buf
           in pass 3 instead of recomputing centering.
        """
        import re

        # Find all reduction indices
        red_indices = [i for i, (kind, _, _) in enumerate(entries) if kind == "reduce"]

        if not red_indices:
            return entries

        last_red_idx = red_indices[-1]

        # Try two-reduction optimization first (LayerNorm pattern)
        if len(red_indices) >= 2 and use_tg_cache:
            result = MetalBackend._optimize_two_reduction_entries(
                entries, red_indices, use_tg_cache=use_tg_cache
            )
            if result is not None:
                return result

        # --- Single-reduction optimization (softmax pattern) ---

        # Collect column vars defined before the last reduction
        pre_red_col_vars: set[str] = set()
        for i in range(last_red_idx):
            kind, _, var_name = entries[i]
            if kind == "col" and var_name is not None:
                pre_red_col_vars.add(var_name)

        if not pre_red_col_vars:
            return entries

        # Find the post-reduction block: column entries after last reduction
        post_entries = entries[last_red_idx + 1 :]
        post_col = [
            (i + last_red_idx + 1, e)
            for i, e in enumerate(post_entries)
            if e[0] == "col"
        ]

        if len(post_col) < 2:
            return entries

        # Check last entry is a store: out[_row * _RDIM + _j] = var;
        last_idx, last_entry = post_col[-1]
        last_line = last_entry[1]
        assert isinstance(last_line, str)
        store_match = re.match(r"(\w+)\[_row \* _RDIM \+ _j\] = (\w+);", last_line)
        if store_match is None:
            return entries

        out_buf = store_match.group(1)
        stored_var = store_match.group(2)

        # Check that the stored value depends on pre-reduction column vars
        # (i.e., ensure_deps would pull them in)
        post_var_lines: dict[str, str] = {}
        for _, entry in post_col[:-1]:
            if entry[2] is not None:
                assert isinstance(entry[1], str)
                post_var_lines[entry[2]] = entry[1]

        # Trace dependencies: does stored_var reach pre-reduction vars?
        def reaches_pre_red(var: str, visited: set[str] | None = None) -> bool:
            if visited is None:
                visited = set()
            if var in visited:
                return False
            visited.add(var)
            if var in pre_red_col_vars:
                return True
            line = post_var_lines.get(var, "")
            for pv in pre_red_col_vars:
                if re.search(rf"\b{re.escape(pv)}\b", line):
                    return True
            # Check transitive deps
            for other_var in post_var_lines:
                if other_var != var and re.search(rf"\b{re.escape(other_var)}\b", line):
                    if reaches_pre_red(other_var, visited):
                        return True
            return False

        if not reaches_pre_red(stored_var):
            return entries

        # Find the last column var in the block immediately before the
        # last reduction (this is the value being reduced AND what we'll
        # store early to the output buffer).
        pre_block_last_var = None
        for i in range(last_red_idx - 1, -1, -1):
            kind, _, var_name = entries[i]
            if kind == "col" and var_name is not None:
                pre_block_last_var = var_name
                break
            if kind != "col":
                break

        if pre_block_last_var is None:
            return entries

        # === Apply the optimization ===
        new_entries = list(entries)

        # Find the reduction entry to get the reduction variable name
        # (before any insertions shift indices)
        red_var = entries[last_red_idx][1]
        assert isinstance(red_var, tuple)
        red_name = red_var[0]  # e.g., "sum_exp"

        if use_tg_cache:
            # TG cache path: store computed intermediate to _tg_row
            # (overwrites raw input, which is no longer needed).
            store_line = f"_tg_row[_j] = {pre_block_last_var};"
            new_entries.insert(last_red_idx, ("col", store_line, None))
            last_red_idx += 1
        else:
            # Device memory path: store intermediate to out_buf before
            # reduction, load it back after.
            store_line = f"{out_buf}[_row * _RDIM + _j] = {pre_block_last_var};"
            new_entries.insert(last_red_idx, ("col", store_line, None))
            last_red_idx += 1

        new_post_start = last_red_idx + 1
        new_entries = new_entries[:new_post_start]

        # Detect the operator applied in the post-reduction block
        if stored_var not in post_var_lines:
            return entries

        expr = post_var_lines[stored_var]
        div_match = re.search(
            rf"\({re.escape(pre_block_last_var)} / {re.escape(red_name)}\)",
            expr,
        )
        mul_match = re.search(
            rf"\({re.escape(pre_block_last_var)} \* {re.escape(red_name)}\)",
            expr,
        )

        if use_tg_cache:
            # TG cache path: read from _tg_row (already loaded), no
            # intermediate device store needed.  _tg_row is row-local
            # (indexed 0..RDIM-1), so use _j directly.
            cached_load = (
                "col",
                "auto _cached = _tg_row[_j];",
                "_cached",
            )
        else:
            # Device memory path: read back from out_buf
            cached_load = (
                "col",
                f"auto _cached = {out_buf}[_row * _RDIM + _j];",
                "_cached",
            )

        # Use multiply-by-reciprocal for division (4x faster on Apple GPU ALUs)
        if div_match:
            new_entries.append(
                ("scalar", f"    float _inv_{red_name} = (1.0f / {red_name});", None)
            )
            update_store = (
                "col",
                f"{out_buf}[_row * _RDIM + _j] = (_cached * _inv_{red_name});",
                None,
            )
        elif mul_match:
            update_store = (
                "col",
                f"{out_buf}[_row * _RDIM + _j] = (_cached * {red_name});",
                None,
            )
        else:
            return entries

        new_entries.append(cached_load)
        new_entries.append(update_store)

        return new_entries

    @staticmethod
    def _optimize_two_reduction_entries(
        entries: list[tuple[str, str | tuple[str, str], str | None]],
        red_indices: list[int],
        *,
        use_tg_cache: bool = False,
    ) -> list[tuple[str, str | tuple[str, str], str | None]] | None:
        """Optimize two-reduction pattern (LayerNorm) with use_tg_cache.

        Pattern:
          col(load x, sum) → reduce1(mean) →
          col(center, square, sum) → reduce2(var) →
          col(re-center, normalize, ...) → store

        Optimization: Store centered values to out_buf in pass 2, read
        from out_buf in pass 3 instead of recomputing centering from
        _tg_row.  Saves 1 TG cache read per pass.
        """
        import re

        if len(red_indices) < 2:
            return None

        red1_idx = red_indices[-2]
        red2_idx = red_indices[-1]

        # Find the output buffer from the store at the end
        out_buf = None
        for i in range(len(entries) - 1, red2_idx, -1):
            kind, data, _ = entries[i]
            if kind == "col" and isinstance(data, str):
                store_match = re.match(r"(\w+)\[_row \* _RDIM \+ _j\] = (\w+);", data)
                if store_match:
                    out_buf = store_match.group(1)
                    break
        if out_buf is None:
            return None

        # Collect column vars defined between red1 and red2
        inter_col_vars: dict[str, int] = {}  # var_name → entry index
        for i in range(red1_idx + 1, red2_idx):
            kind, _, var_name = entries[i]
            if kind == "col" and var_name is not None:
                inter_col_vars[var_name] = i

        if not inter_col_vars:
            return None

        # Find which inter-reduction col vars are referenced in post-red2
        # column entries (these would be pulled in by ensure_deps)
        post_col_text = ""
        for i in range(red2_idx + 1, len(entries)):
            kind, data, _ = entries[i]
            if kind == "col" and isinstance(data, str):
                post_col_text += " " + data

        # Find the inter-reduction var that is directly referenced
        # in the post-red2 block (the var to cache)
        cache_var = None
        for var in inter_col_vars:
            if re.search(rf"\b{re.escape(var)}\b", post_col_text):
                cache_var = var
                break

        if cache_var is None:
            return None

        # Build optimized entries:
        # 1. Keep everything up to (but not including) red2
        # 2. Insert store of cache_var to out_buf before red2
        # 3. Keep red2
        # 4. Keep scalar entries after red2
        # 5. Replace post-red2 col entries: load from out_buf, rewrite refs

        new_entries = list(entries[:red2_idx])

        # Insert store of centered values to out_buf before red2, then red2
        new_entries.extend(
            [
                (
                    "col",
                    f"{out_buf}[_row * _RDIM + _j] = {cache_var};",
                    None,
                ),
                entries[red2_idx],
            ]
        )

        # Process post-red2 entries
        loaded_cached = False
        for i in range(red2_idx + 1, len(entries)):
            kind, data, var_name = entries[i]
            if kind == "scalar":
                new_entries.append((kind, data, var_name))
            elif kind == "col" and isinstance(data, str):
                if not loaded_cached:
                    # Insert load from out_buf as first col entry
                    new_entries.append(
                        (
                            "col",
                            f"auto _cached = {out_buf}[_row * _RDIM + _j];",
                            "_cached",
                        )
                    )
                    loaded_cached = True
                # Rewrite references to cache_var with _cached
                new_data = re.sub(rf"\b{re.escape(cache_var)}\b", "_cached", data)
                # If this was the definition of cache_var, skip it
                # (it's now loaded from out_buf)
                if var_name == cache_var:
                    continue
                new_var = var_name
                if new_var == cache_var:
                    new_var = "_cached"
                new_entries.append(("col", new_data, new_var))
            else:
                new_entries.append((kind, data, var_name))

        return new_entries

    @staticmethod
    def _emit_reduction_body(
        msl_parts: list[str],
        entries: list[tuple[str, str | tuple[str, str], str | None]],
        buf_1d: dict[str, str],
        *,
        tg_row_cache: dict[str, str] | None = None,
    ) -> None:
        """Emit generic reduction MSL from pre-classified entries.

        Handles any combination of reductions (max, sum, min) with
        arbitrary elementwise ops between and after them.  One threadgroup
        per row, all threads cooperate via SIMD shuffle + shared memory.

        ``entries`` is a list of (kind, data, var_name) tuples from
        ``_classify_reduction_stmts``.  ``buf_1d`` maps local names to
        their 1D broadcast source buffer names.
        """
        import re

        if tg_row_cache is None:
            tg_row_cache = {}
        use_tg_cache = bool(tg_row_cache)

        # Optimize: fuse store into reduction pass to eliminate redundant
        # recomputation (e.g., 3-pass softmax → 2-pass).
        entries = MetalBackend._optimize_reduction_entries(
            entries, use_tg_cache=use_tg_cache
        )

        msl_parts.extend(
            [
                "    uint _row = _tg_pos;",
                "    if (_row >= _NROWS) return;",
                "    uint _RDIM4 = _RDIM & ~3u;",
                "",
            ]
        )

        # Emit threadgroup row cache buffer and cooperative load
        if tg_row_cache:
            for input_buf, tg_name in tg_row_cache.items():
                msl_parts.extend(
                    [
                        f"    threadgroup float {tg_name}[_RDIM];",
                        "    for (uint _ci = _tid; _ci < _RDIM; _ci += _tg_size)",
                        f"        {tg_name}[_ci] = {input_buf}[_row * _RDIM + _ci];",
                        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                        "",
                    ]
                )

        num_reductions = sum(1 for k, _, _ in entries if k == "reduce")
        if num_reductions > 0:
            # Each reduction needs its own shared memory array to avoid
            # cross-reduction races on Apple GPUs.  When a second reduction
            # writes to the same _shared[_sg] that a previous reduction
            # broadcast via _shared[0], some threads may still observe the
            # stale first-reduction value despite the threadgroup barrier.
            for ri in range(num_reductions):
                msl_parts.append(f"    threadgroup float _shared_{ri}[32];")
            msl_parts.extend(
                [
                    "    uint _lane = _tid % 32;",
                    "    uint _sg = _tid / 32;",
                    "    uint _num_sg = (_tg_size + 31) / 32;",
                    "",
                ]
            )
        _reduction_counter = [0]  # mutable counter for emit_simd_reduce

        def fixup(line: str, *, vec_lane: int | None = None) -> str:
            line = line.replace("_gid * _RDIM", "_row * _RDIM")
            for lv, bn in buf_1d.items():
                if vec_lane is not None:
                    line = line.replace(lv, f"_ld1d_{bn}.{comps[vec_lane]}")
                else:
                    line = line.replace(lv, f"{bn}[_j]")
            return line

        comps = "xyzw"

        def _emit_red_accum(red_var: str, red_op: str, val: str) -> None:
            if red_op == "max":
                msl_parts.append(f"        {red_var} = max({red_var}, (float){val});")
            elif red_op == "sum":
                msl_parts.append(f"        {red_var} += (float){val};")
            else:
                msl_parts.append(f"        {red_var} = min({red_var}, (float){val});")

        # Patterns for detecting buffer loads/stores after fixup
        _LOAD_RE = re.compile(r"auto (\w+) = (\w+)\[(_row \* _RDIM) \+ _j\];")
        _LOAD_1D_RE = re.compile(r"auto (\w+) = (\w+)\[_j\];")
        _STORE_RE = re.compile(r"(\w+)\[(_row \* _RDIM) \+ _j\] = (.+);")
        _STORE_MUL_RE = re.compile(r"(\w+)\[(_row \* _RDIM) \+ _j\] \*= (.+);")
        _STORE_1D_RE = re.compile(r"(\w+)\[_j\] = (.+);")

        def flush_col_block(
            block: list[tuple[str, str | None]],
            red: tuple[str, str] | None,
        ) -> None:
            if not block and red is None:
                return

            red_var = red_op = None
            if red is not None:
                red_var, red_op = red
                identity = {
                    "max": "-INFINITY",
                    "sum": "0.0f",
                    "min": "INFINITY",
                }[red_op]
                msl_parts.append(f"    float {red_var} = {identity};")

            # Collect variable names declared in this block
            block_vars = [vn for _, vn in block if vn is not None]
            last_var_in_block = block_vars[-1] if block_vars else None

            # --- Vectorized main loop: float4 loads/stores ---
            msl_parts.append(
                "    for (uint _jv = _tid; _jv < _RDIM4 / 4; _jv += _tg_size) {"
            )

            # Emit float4 loads for 1D broadcast buffers used in block
            block_text = " ".join(ln for ln, _ in block)
            for lv, bn in buf_1d.items():
                if lv in block_text:
                    msl_parts.append(
                        f"        float4 _ld1d_{bn} = ((device float4*){bn})[_jv];"
                    )

            # Names that live in threadgroup address space
            tg_names = set(tg_row_cache.values()) if tg_row_cache else set()

            for line, _vn in block:
                fixed = fixup(line)

                # Buffer load: auto VAR = BUF[_row * _RDIM + _j];
                m_ld = _LOAD_RE.match(fixed)
                if m_ld:
                    var, buf, base = m_ld.groups()
                    if buf in tg_names:
                        # TG-resident: row-local indexing, threadgroup space
                        msl_parts.append(
                            f"        float4 _ld_{var} = "
                            f"((threadgroup float4*){buf})[_jv];"
                        )
                    elif buf in (tg_row_cache or {}):
                        # Input buffer that has a TG cache — read from cache
                        tg_name = tg_row_cache[buf]
                        msl_parts.append(
                            f"        float4 _ld_{var} = "
                            f"((threadgroup float4*){tg_name})[_jv];"
                        )
                    else:
                        msl_parts.append(
                            f"        float4 _ld_{var} = "
                            f"((device float4*)({buf} + {base}))[_jv];"
                        )
                    for k in range(4):
                        msl_parts.append(
                            f"        auto {var}_{k} = _ld_{var}.{comps[k]};"
                        )
                    continue

                # 1D broadcast load: auto VAR = BUF[_j];
                m_1d = _LOAD_1D_RE.match(fixed)
                if m_1d:
                    var, buf = m_1d.groups()
                    addr_space = "threadgroup" if buf in tg_names else "device"
                    msl_parts.append(
                        f"        float4 _ld_{var} = (({addr_space} float4*){buf})[_jv];"
                    )
                    for k in range(4):
                        msl_parts.append(
                            f"        auto {var}_{k} = _ld_{var}.{comps[k]};"
                        )
                    continue

                # Store: BUF[_row * _RDIM + _j] = EXPR;
                m_st = _STORE_RE.match(fixed)
                if m_st:
                    buf, base, expr = m_st.groups()
                    # Check if expr references any block var
                    refs_block = any(
                        re.search(rf"\b{re.escape(bv)}\b", expr) for bv in block_vars
                    )
                    if refs_block:
                        lane_exprs = []
                        for k in range(4):
                            e = expr
                            for bv in block_vars:
                                e = re.sub(
                                    rf"\b{re.escape(bv)}\b",
                                    f"{bv}_{k}",
                                    e,
                                )
                            lane_exprs.append(e)
                        msl_parts.append(
                            f"        ((device float4*)({buf} + {base}))"
                            f"[_jv] = float4("
                            f"{', '.join(lane_exprs)});"
                        )
                    else:
                        # Scalar broadcast store
                        msl_parts.append(
                            f"        ((device float4*)({buf} + {base}))"
                            f"[_jv] = float4({expr});"
                        )
                    continue

                # Multiply-assign store: BUF[...] *= EXPR;
                m_mul = _STORE_MUL_RE.match(fixed)
                if m_mul:
                    buf, base, expr = m_mul.groups()
                    msl_parts.append(
                        f"        ((device float4*)({buf} + {base}))[_jv] *= {expr};"
                    )
                    continue

                # 1D store: BUF[_j] = EXPR; (threadgroup row cache)
                m_st1d = _STORE_1D_RE.match(fixed)
                if m_st1d:
                    buf, expr = m_st1d.groups()
                    addr_space = "threadgroup" if buf in tg_names else "device"
                    refs_block = any(
                        re.search(rf"\b{re.escape(bv)}\b", expr) for bv in block_vars
                    )
                    if refs_block:
                        lane_exprs = []
                        for k in range(4):
                            e = expr
                            for bv in block_vars:
                                e = re.sub(
                                    rf"\b{re.escape(bv)}\b",
                                    f"{bv}_{k}",
                                    e,
                                )
                            lane_exprs.append(e)
                        msl_parts.append(
                            f"        (({addr_space} float4*){buf})"
                            f"[_jv] = float4("
                            f"{', '.join(lane_exprs)});"
                        )
                    else:
                        msl_parts.append(
                            f"        (({addr_space} float4*){buf})"
                            f"[_jv] = float4({expr});"
                        )
                    continue

                # Compute line: emit 4 scalar copies
                for k in range(4):
                    lane = fixup(line, vec_lane=k)
                    lane = lane.replace("_gid * _RDIM", "_row * _RDIM")
                    for bv in block_vars:
                        lane = re.sub(rf"\b{re.escape(bv)}\b", f"{bv}_{k}", lane)
                    msl_parts.append(f"        {lane}")

            if red_var is not None and last_var_in_block is not None:
                for k in range(4):
                    _emit_red_accum(red_var, red_op, f"{last_var_in_block}_{k}")

            msl_parts.append("    }")

            # --- Scalar tail for _RDIM % 4 remainder ---
            msl_parts.append(
                "    for (uint _j = _RDIM4 + _tid; _j < _RDIM; _j += _tg_size) {"
            )
            last_var: str | None = None
            for line, vn in block:
                fixed = fixup(line)
                # Rewrite device buffer loads to TG cache in scalar tail
                if tg_row_cache:
                    for dev_buf, tg_name in tg_row_cache.items():
                        fixed = fixed.replace(
                            f"{dev_buf}[_row * _RDIM + _j]",
                            f"{tg_name}[_j]",
                        )
                msl_parts.append(f"        {fixed}")
                if vn is not None:
                    last_var = vn

            if red_var is not None and last_var is not None:
                _emit_red_accum(red_var, red_op, last_var)

            msl_parts.append("    }")

        def emit_simd_reduce(var: str, op: str) -> None:
            ri = _reduction_counter[0]
            _reduction_counter[0] += 1
            sh = f"_shared_{ri}"
            ops = {
                "max": (
                    f"{var} = max({var}, simd_shuffle_down({var}, _off))",
                    "-INFINITY",
                    "_v = max(_v, simd_shuffle_down(_v, _off))",
                ),
                "sum": (
                    f"{var} += simd_shuffle_down({var}, _off)",
                    "0.0f",
                    "_v += simd_shuffle_down(_v, _off)",
                ),
                "min": (
                    f"{var} = min({var}, simd_shuffle_down({var}, _off))",
                    "INFINITY",
                    "_v = min(_v, simd_shuffle_down(_v, _off))",
                ),
            }
            shuffle_op, identity, cross_sg = ops[op]
            msl_parts.extend(
                [
                    f"    for (uint _off = 16; _off > 0; _off >>= 1) {shuffle_op};",
                    f"    if (_lane == 0) {sh}[_sg] = {var};",
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                    "    if (_sg == 0) {",
                    f"        float _v = (_lane < _num_sg) ? {sh}[_lane] : {identity};",
                    f"        for (uint _off = 16; _off > 0; _off >>= 1) {cross_sg};",
                    f"        {sh}[0] = _v;",
                    "    }",
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                    f"    {var} = {sh}[0];",
                    "",
                ]
            )

        # All column-live lines seen so far (var_name → (msl_line, var_name))
        all_col_lines: dict[str, tuple[str, str | None]] = {}

        def ensure_deps(
            block: list[tuple[str, str | None]],
        ) -> list[tuple[str, str | None]]:
            defined: set[str] = set()
            for _, vn in block:
                if vn is not None:
                    defined.add(vn)

            needed: dict[str, tuple[str, str | None]] = {}
            changed = True
            check_vars = set(all_col_lines.keys()) - defined
            while changed:
                changed = False
                for var in list(check_vars):
                    if var in needed:
                        continue
                    entry = all_col_lines[var]
                    all_text = [ln for ln, _ in block] + [
                        ln for ln, _ in needed.values()
                    ]
                    if any(re.search(rf"\b{re.escape(var)}\b", bl) for bl in all_text):
                        needed[var] = entry
                        changed = True

            ordered = [entry for var, entry in all_col_lines.items() if var in needed]
            return ordered + block

        col_block: list[tuple[str, str | None]] = []
        for kind, data, var_name in entries:
            if kind == "col":
                assert isinstance(data, str)
                if var_name is not None:
                    all_col_lines[var_name] = (data, var_name)
                col_block.append((data, var_name))
            elif kind == "reduce":
                assert isinstance(data, tuple)
                flush_col_block(ensure_deps(col_block), red=data)
                col_block = []
                emit_simd_reduce(data[0], data[1])
            elif kind == "scalar":
                if col_block:
                    flush_col_block(ensure_deps(col_block), red=None)
                    col_block = []
                assert isinstance(data, str)
                msl_parts.append(data)

        if col_block:
            flush_col_block(ensure_deps(col_block), red=None)
            col_block = []

    def supports_config_key(self, key: str) -> bool:
        return key in _METAL_SUPPORTED_KEYS

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from ..autotuner.config_fragment import BooleanFragment

        return {"use_tg_cache": BooleanFragment()}

    def adjust_block_size_constraints(
        self, block_specs: list[object], ndim: int
    ) -> None:
        from ..autotuner.config_spec import BlockSizeSpec

        for spec in block_specs:
            if isinstance(spec, BlockSizeSpec):
                spec.update_min(METAL_SIMD_WIDTH)
        # For matmul kernels (3+ block specs), the last block spec is the
        # K-tile. When block_sizes[2] >= K, the descriptor uses a static K
        # (MPP can unroll). When < K, it uses dynamic_length_v (MPP
        # pipelines internally). The autotuner searches both strategies.
        if len(block_specs) >= 3:
            last = block_specs[-1]
            if isinstance(last, BlockSizeSpec):
                last.update_min(METAL_SIMD_WIDTH)

    def pin_num_warps(self, ndim: int) -> int | None:
        # num_warps controls simdgroup count for all kernel types:
        # threads_per_threadgroup = num_warps * 32.
        return None

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        from ..autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        from ..autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def supports_precompile(self) -> bool:
        return False

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        if isinstance(err, Exception):
            return "debug"
        return None

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        bound_kernel.settings.autotune_precompile = None
        return super().autotune(bound_kernel, args, force=force, **kwargs)

    # --- launcher args ---

    @staticmethod
    def _resolve_name(
        name: str,
        arg_map: dict[str, object],
        load_to_arg: dict[str, str],
    ) -> object | None:
        """Resolve a name to a TensorArg via arg_map and load indirection.

        Used by both MslWalker._resolve_arg and _resolve_matmul_dims.
        """
        if name in arg_map:
            return arg_map[name]
        src = load_to_arg.get(name, name)
        return arg_map.get(src)

    @staticmethod
    def _resolve_matmul_dims(
        mm_op: MetalMatmulOp,
        arg_map: dict[str, object],
        load_to_arg: dict[str, str],
    ) -> tuple[int, int, int, bool, int, object, object]:
        """Resolve matmul dimensions from structured IR + tensor args.

        Returns (M, N, K, is_batched, batch, lhs_arg, rhs_arg).
        """
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        env = CompileEnvironment.current()

        lhs_obj = MetalBackend._resolve_name(mm_op.lhs_name, arg_map, load_to_arg)
        rhs_obj = MetalBackend._resolve_name(mm_op.rhs_name, arg_map, load_to_arg)
        lhs_arg = lhs_obj if isinstance(lhs_obj, TensorArg) else None
        rhs_arg = rhs_obj if isinstance(rhs_obj, TensorArg) else None
        assert lhs_arg is not None, f"Cannot resolve LHS arg: {mm_op.lhs_name}"
        assert rhs_arg is not None, f"Cannot resolve RHS arg: {mm_op.rhs_name}"

        is_batched = lhs_arg.fake_value.ndim >= 3
        batch = env.size_hint(lhs_arg.fake_value.size(0)) if is_batched else 1
        M = env.size_hint(lhs_arg.fake_value.size(-2))
        K = env.size_hint(lhs_arg.fake_value.size(-1))
        rhs_d0 = env.size_hint(rhs_arg.fake_value.size(-2))
        rhs_d1 = env.size_hint(rhs_arg.fake_value.size(-1))
        N = rhs_d1 if rhs_d0 == K else rhs_d0

        return M, N, K, is_batched, batch, lhs_arg, rhs_arg

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
        device_fn: DeviceFunction | None = None,
    ) -> list[str]:
        """Build Metal launcher arguments.

        Classifies the kernel from ``_metal_ops`` (populated during
        lowering, before this method is called) and resolves dimensions
        from full tensor arg shapes via ``_resolve_matmul_dims``.

        Note: ``build_launcher_args`` is called BEFORE
        ``generate_msl_function``, so we classify independently here.
        """
        from .device_function import TensorArg

        out = [*args]

        # Classify from structured IR
        kind = self._classify_kernel()

        if kind == MetalKernelKind.FUSED_ATTENTION and sorted_args is not None:
            assert device_fn is not None
            mm_ops = [op for op in self._metal_ops if isinstance(op, MetalMatmulOp)]
            first_mm = mm_ops[0]
            arg_map: dict[str, object] = {
                a.name: a for a in sorted_args if isinstance(a, TensorArg)
            }
            load_to_arg: dict[str, str]
            load_to_arg, _ = MetalBackend._build_arg_maps(
                device_fn.preamble + device_fn.body, arg_map
            )
            M_val, N1_val, _, is_batched, batch_val, _, _ = self._resolve_matmul_dims(
                first_mm, arg_map, load_to_arg
            )

            tile_m_idx = 1 if is_batched and len(config.block_sizes) > 1 else 0
            TILE_M = (
                config.block_sizes[tile_m_idx]
                if len(config.block_sizes) > tile_m_idx
                else 64
            )
            num_tiles = (M_val + TILE_M - 1) // TILE_M
            NUM_SG = config.num_warps if config.num_warps is not None else 4
            tpg = METAL_SIMD_WIDTH * NUM_SG

            # Unified walker always uses MPP cooperative_tensor with scratch
            scratch_size = batch_val * M_val * N1_val
            grid_y = (num_tiles + NUM_SG - 1) // NUM_SG
            out.extend(
                [
                    f"_block_size={tpg}",
                    f"_composed_grid={grid_y}",
                    f"_scratch_size={scratch_size}",
                    f"_num_simdgroups={NUM_SG}",
                    f"_batch_size={batch_val}",
                ]
            )

        elif kind == MetalKernelKind.MATMUL and sorted_args is not None:
            assert device_fn is not None
            mm_ops = [op for op in self._metal_ops if isinstance(op, MetalMatmulOp)]
            first_mm = mm_ops[0]
            arg_map_mm: dict[str, object] = {
                a.name: a for a in sorted_args if isinstance(a, TensorArg)
            }
            load_to_arg_mm: dict[str, str]
            load_to_arg_mm, _ = MetalBackend._build_arg_maps(
                device_fn.preamble + device_fn.body, arg_map_mm
            )
            M, N, _, _, _, _, _ = self._resolve_matmul_dims(
                first_mm, arg_map_mm, load_to_arg_mm
            )
            TILE_M = config.block_sizes[0] if len(config.block_sizes) > 0 else 64
            TILE_N = config.block_sizes[1] if len(config.block_sizes) > 1 else 32
            NUM_SG = config.num_warps if config.num_warps is not None else 4
            grid_m = (M + TILE_M - 1) // TILE_M
            grid_n = (N + TILE_N - 1) // TILE_N
            tpg = METAL_SIMD_WIDTH * NUM_SG
            out.extend(
                [
                    f"_block_size={tpg}",
                    f"_matmul_grid=({grid_m}, {grid_n})",
                    f"_num_simdgroups={NUM_SG}",
                ]
            )

        elif kind == MetalKernelKind.SOFTMAX:
            NUM_SG = config.num_warps if config.num_warps is not None else 4
            tpg = METAL_SIMD_WIDTH * NUM_SG
            out.append(f"_block_size={tpg}")
            if sorted_args is not None:
                for arg in sorted_args:
                    if isinstance(arg, TensorArg) and arg.fake_value.ndim >= 2:
                        nrows_expr = f"{tensor_host_args[0]}.size(0)"
                        out.append(f"_nrows={nrows_expr}")
                        break

        else:  # ELEMENTWISE
            block_size = config.block_sizes[0] if config.block_sizes else 256
            out.append(f"_block_size={block_size}")

        return out
