from __future__ import annotations

import abc
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
    }
)


class MetalKernelKind:
    """Classification of a Metal kernel for dispatch and codegen."""

    ELEMENTWISE = "elementwise"  # 1D flat dispatch
    SOFTMAX = "softmax"  # 2D row-parallel (one threadgroup per row)
    MATMUL = "matmul"  # 2D MPP matmul2d
    FUSED_ATTENTION = "fused_attention"  # matmul + softmax + matmul


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
            return f"_metal_{reduction_type}({input_name}, {dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    # --- MSL assembly ---

    def generate_msl_function(self, device_fn: DeviceFunction) -> list[ast.stmt]:
        """Build a Python function that returns ``(msl_source, kernel_name)``.

        Called from ``DeviceFunction.codegen_function_def()`` when the
        backend is Metal.  Collects argument metadata, unparses the body
        AST to pseudo-MSL text, and wraps everything in a complete MSL
        kernel string.
        """
        import ast as _ast

        from .ast_extension import create
        from .ast_extension import create_arguments
        from .ast_extension import statement_from_string
        from .device_function import TensorArg

        kernel_name = device_fn.name

        # --- build MSL source pieces ---
        msl_parts: list[str] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]

        from .device_function import ConstExprArg
        from .device_function import NumericArgument

        # Kernel signature: only tensor args become Metal buffer params.
        # Constexpr/scalar args are baked into the MSL as constants.
        params: list[str] = []
        buf_idx = 0
        constexpr_defines: list[str] = []
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                metal_dtype = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {metal_dtype}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
            elif isinstance(arg, (ConstExprArg, NumericArgument)):
                # Bake scalar value directly into MSL as a constant
                constexpr_defines.append(f"constant int {arg.name} = {arg.host_str()};")

        # Global thread ID for 1D per-thread dispatch
        params.append("uint _gid [[thread_position_in_grid]]")

        # Emit constexpr constants before the kernel function
        msl_parts.extend(constexpr_defines)

        sig = ", ".join(params)
        msl_parts.append(f"kernel void {kernel_name}({sig}) {{")

        # Unparse body AST to Python text, then transform to MSL
        msl_body_lines: list[str] = []
        for stmt in device_fn.preamble + device_fn.body:
            py_text = _ast.unparse(stmt).strip()
            msl_line = self._py_to_msl(py_text)
            msl_body_lines.append(msl_line)

        # Classify kernel from body sentinels (single source of truth).
        body_text = "\n".join(msl_body_lines)
        has_matmul = (
            "_metal_addmm" in body_text
            or "_metal_mm" in body_text
            or "_metal_bmm" in body_text
            or "_metal_baddbmm" in body_text
        )
        has_reduction = "_RDIM" in body_text

        if has_matmul and has_reduction:
            self._kernel_kind = MetalKernelKind.FUSED_ATTENTION
        elif has_matmul:
            self._kernel_kind = MetalKernelKind.MATMUL
        elif has_reduction:
            self._kernel_kind = MetalKernelKind.SOFTMAX
        else:
            self._kernel_kind = MetalKernelKind.ELEMENTWISE

        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()

        if self._kernel_kind == MetalKernelKind.FUSED_ATTENTION:
            # Validate: fused attention requires both matmul and reduction
            import re as _re

            matmul_count = len(
                _re.findall(r"_metal_(?:b?mm|addmm|baddbmm)\(", body_text)
            )
            if matmul_count < 2:
                raise exc.BackendUnsupported(
                    "metal",
                    f"matmul+reduction kernel has {matmul_count} matmul(s), "
                    "but fused attention requires at least 2 (scores + output). "
                    "Single matmul + reduction is not yet supported.",
                )
            self._emit_fused_attention_body(
                msl_parts, msl_body_lines, device_fn, params, env
            )
        elif self._kernel_kind == MetalKernelKind.MATMUL:
            self._emit_matmul_body(msl_parts, msl_body_lines, device_fn, params, env)
        elif has_reduction:
            # Infer _RDIM and _NROWS from the first 2D+ tensor
            rdim_val = 1
            nrows_val = 1
            for arg in device_fn.sorted_args():
                if isinstance(arg, TensorArg) and arg.fake_value.ndim >= 2:
                    rdim_val = env.size_hint(arg.fake_value.size(-1))
                    nrows_val = env.size_hint(arg.fake_value.size(0))
                    break

            # Insert constants before the kernel function
            insert_pos = len(constexpr_defines) + 3  # after includes + blank
            msl_parts.insert(insert_pos, f"constant uint _RDIM = {rdim_val};")
            msl_parts.insert(insert_pos + 1, f"constant uint _NROWS = {nrows_val};")

            # Replace _gid with threadgroup-parallel params
            params.pop()  # remove "uint _gid [[thread_position_in_grid]]"
            params.extend(
                [
                    "uint _tg_pos [[threadgroup_position_in_grid]]",
                    "uint _tid [[thread_position_in_threadgroup]]",
                    "uint _tg_size [[threads_per_threadgroup]]",
                ]
            )

            self._emit_reduction_body(msl_parts, msl_body_lines)
        else:
            for line in msl_body_lines:
                msl_parts.append(f"    {line}")

        msl_parts.append("}")
        # Re-generate signature since params may have been modified
        sig = ", ".join(params)
        # Replace the kernel signature line
        for i, part in enumerate(msl_parts):
            if part.startswith("kernel void"):
                msl_parts[i] = f"kernel void {kernel_name}({sig}) {{"
                break
        msl_source = "\n".join(msl_parts)

        # Build a Python function that returns (msl_source, kernel_name)
        # The function takes constexpr args so block sizes can be baked in.
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

    @staticmethod
    def _emit_reduction_body(msl_parts: list[str], body_lines: list[str]) -> None:
        """Emit generic reduction MSL from body lines.

        Handles any combination of reductions (max, sum, min) with
        arbitrary elementwise ops between and after them.  One threadgroup
        per row, all threads cooperate via SIMD shuffle + shared memory.

        The body lines from ``_py_to_msl()`` use sentinels:
          - ``_reduce_max_val`` / ``_reduce_sum_val`` / ``_reduce_min_val``
          - ``_gid * _RDIM + _j`` for 2D row-column indexing
          - ``buf[:, :]`` for 1D weight/bias loads

        Algorithm:
        1. Parse lines into an ordered list of ``(kind, data)`` entries
           where kind is ``"col"`` (per-column), ``"scalar"`` (row-wide),
           ``"reduce"`` (reduction sentinel), or ``"load1d"`` (weight load).
        2. Walk the entries tracking which variables are "column-live" (defined
           inside a column loop and not yet reduced to a scalar).  A line that
           references any column-live variable is itself columnar.
        3. Group consecutive columnar entries into loops.  Each reduction
           becomes a SIMD shuffle + shared mem reduce of an accumulator
           that was fed in the preceding column loop.
        """
        import re

        _REDUCE_RE = re.compile(r"auto (\w+) = .*_reduce_(max|sum|min)_val.*")
        _LOAD_1D_RE = re.compile(r"auto (\w+) = (\w+)\[:,\s*:\];")
        _FULL_RE = re.compile(
            r"auto (\w+) = (?:static_cast<\w+>\()?tl\.full\(\[\],\s*([^,]+),\s*tl\.\w+\)?;"
        )
        _ASSIGN_RE = re.compile(r"(?:auto )?(\w+)\s*=\s*(.+);")
        _COL_MARKER = re.compile(r"_gid \* _RDIM \+ _j|_RDIM|\b_j\b")

        # 1D buffer substitutions: load_var -> buf_name
        buf_1d: dict[str, str] = {}

        # --- Step 1: Parse into entries ---
        Entry = tuple  # (kind, payload)
        entries: list[Entry] = []

        for raw_line in body_lines:
            line = raw_line.strip()
            if not line or line.startswith("auto indices_"):
                continue

            # Strip tl.reshape wrappers
            line = re.sub(
                r"static_cast<(\w+)>\(tl\.reshape\(([^)]+)\),\s*\[[^\]]*\]\)",
                r"static_cast<\1>(\2)",
                line,
            )
            line = re.sub(r"tl\.reshape\(([^,]+),\s*\[[^\]]*\]\)", r"\1", line)

            m = _REDUCE_RE.match(line)
            if m:
                entries.append(("reduce", (m.group(1), m.group(2))))
                continue

            m = _LOAD_1D_RE.match(line)
            if m:
                buf_1d[m.group(1)] = m.group(2)
                continue

            m = _FULL_RE.match(line)
            if m:
                entries.append(("scalar", f"float {m.group(1)} = {m.group(2)};"))
                continue

            entries.append(("line", line))

        # --- Step 2: Determine column-liveness ---
        # A variable is "column-live" if it was defined in a line that
        # directly touches _j/_RDIM, or that references another col-live var.
        col_live: set[str] = set()  # names of per-column variables

        def is_col_line(line: str) -> bool:
            if _COL_MARKER.search(line):
                return True
            return any(
                re.search(rf"\b{re.escape(v)}\b", line) for v in col_live
            )

        # Classify each "line" entry as col or scalar
        classified: list[Entry] = []
        for kind, data in entries:
            if kind == "line":
                if is_col_line(data):
                    # Track the LHS as column-live
                    m = _ASSIGN_RE.match(data)
                    if m:
                        col_live.add(m.group(1))
                    classified.append(("col", data))
                else:
                    classified.append(("scalar", f"    {data}"))
            elif kind == "reduce":
                var_name, _ = data
                # Reduction output is scalar (row-wide)
                col_live.discard(var_name)
                classified.append(("reduce", data))
            else:
                classified.append((kind, data))

        # --- Step 3: Emit MSL ---
        msl_parts.extend(
            [
                "    uint _row = _tg_pos;",
                "    if (_row >= _NROWS) return;",
                "",
            ]
        )

        has_reductions = any(k == "reduce" for k, _ in classified)
        if has_reductions:
            msl_parts.extend(
                [
                    "    threadgroup float _shared[32];",
                    "    uint _lane = _tid % 32;",
                    "    uint _sg = _tid / 32;",
                    "    uint _num_sg = (_tg_size + 31) / 32;",
                    "",
                ]
            )

        def fixup(line: str) -> str:
            """Apply row/column substitutions to a body line."""
            line = line.replace("_gid * _RDIM", "_row * _RDIM")
            for lv, bn in buf_1d.items():
                line = line.replace(lv, f"{bn}[_j]")
            return line

        def flush_col_block(block: list[str], red: tuple[str, str] | None) -> None:
            """Emit a column loop from *block*, optionally accumulating *red*."""
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

            msl_parts.append(
                "    for (uint _j = _tid; _j < _RDIM; _j += _tg_size) {"
            )
            # Find the last assigned variable in the block — it feeds the reduction
            last_var: str | None = None
            for line in block:
                msl_parts.append(f"        {fixup(line)}")
                m = _ASSIGN_RE.match(line)
                if m:
                    last_var = m.group(1)

            if red_var is not None and last_var is not None:
                if red_op == "max":
                    msl_parts.append(
                        f"        {red_var} = max({red_var}, (float){last_var});"
                    )
                elif red_op == "sum":
                    msl_parts.append(f"        {red_var} += (float){last_var};")
                else:
                    msl_parts.append(
                        f"        {red_var} = min({red_var}, (float){last_var});"
                    )

            msl_parts.append("    }")

        def emit_simd_reduce(var: str, op: str) -> None:
            """Emit SIMD shuffle + shared mem reduction."""
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
                    f"    if (_lane == 0) _shared[_sg] = {var};",
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                    "    if (_sg == 0) {",
                    f"        float _v = (_lane < _num_sg) ? _shared[_lane] : {identity};",
                    f"        for (uint _off = 16; _off > 0; _off >>= 1) {cross_sg};",
                    "        _shared[0] = _v;",
                    "    }",
                    "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                    f"    {var} = _shared[0];",
                    "",
                ]
            )

        # Track all columnar lines by the variable they define.
        # When a later loop references a variable from a prior loop,
        # we re-emit that line (and transitively its dependencies).
        all_col_lines: dict[str, str] = {}  # var_name -> line

        def ensure_deps(block: list[str]) -> list[str]:
            """Prepend columnar lines for any referenced-but-not-defined vars."""
            defined: set[str] = set()
            for line in block:
                m = _ASSIGN_RE.match(line)
                if m:
                    defined.add(m.group(1))

            # Iteratively resolve transitive deps
            needed: dict[str, str] = {}
            changed = True
            check_vars = set(all_col_lines.keys()) - defined
            while changed:
                changed = False
                for var in list(check_vars):
                    if var in needed:
                        continue
                    line = all_col_lines[var]
                    # Check if referenced in block or in already-needed lines
                    all_text = block + list(needed.values())
                    if any(
                        re.search(rf"\b{re.escape(var)}\b", bl)
                        for bl in all_text
                    ):
                        needed[var] = line
                        changed = True

            # Topological order: emit needed lines in their original order
            ordered = [
                line
                for var, line in all_col_lines.items()
                if var in needed
            ]
            return ordered + block

        # Walk classified entries, grouping col lines into blocks
        col_block: list[str] = []
        for kind, data in classified:
            if kind == "col":
                # Track all columnar lines by defined variable
                m = _ASSIGN_RE.match(data)
                if m:
                    all_col_lines[m.group(1)] = data
                col_block.append(data)
            elif kind == "reduce":
                # Flush preceding col block as a loop with this reduction
                flush_col_block(ensure_deps(col_block), red=data)
                col_block = []
                emit_simd_reduce(data[0], data[1])
            elif kind == "scalar":
                # Flush any pending col block first (without reduction)
                if col_block:
                    flush_col_block(ensure_deps(col_block), red=None)
                    col_block = []
                msl_parts.append(data)

        # Flush remaining col block (final pass — stores, etc.)
        if col_block:
            flush_col_block(ensure_deps(col_block), red=None)
            col_block = []

    @staticmethod
    def _extract_matmul_args(
        body_lines: list[str],
        device_fn: DeviceFunction,
        env: object,
    ) -> tuple[object, object, object, int, int, int]:
        """Extract matmul buffer args and dimensions from body lines.

        Returns (lhs_arg, rhs_arg, out_arg, M, K, N).
        """
        import re

        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        assert isinstance(env, CompileEnvironment)

        lhs_buf = rhs_buf = out_buf = None
        for line in body_lines:
            # 2-arg matmul: _metal_mm or _metal_bmm
            m = re.search(r"_metal_b?mm\((\w+),\s*(\w+)\)", line)
            if m:
                lhs_buf = m.group(1)
                rhs_buf = m.group(2)
                m2 = re.match(r"auto (\w+)\s*=", line)
                if m2:
                    out_buf = m2.group(1)
            # 3-arg matmul: _metal_addmm or _metal_baddbmm
            for pat in [
                r"_metal_addmm\((\w+),\s*(\w+),\s*(\w+)\)",
                r"_metal_baddbmm\((\w+),\s*(\w+),\s*(\w+)\)",
            ]:
                m = re.search(pat, line)
                if m:
                    lhs_buf = m.group(2)
                    rhs_buf = m.group(3)
                    m2 = re.match(r"auto (\w+)\s*=", line)
                    if m2:
                        out_buf = m2.group(1)
                    break

        if out_buf is None:
            for line in body_lines:
                m = re.match(r"(\w+)\[", line)
                if m and m.group(1) not in (lhs_buf, rhs_buf):
                    out_buf = m.group(1)
                    break

        tensor_args = [a for a in device_fn.sorted_args() if isinstance(a, TensorArg)]
        arg_map: dict[str, TensorArg] = {}
        for arg in tensor_args:
            arg_map[arg.name] = arg

        lhs_arg = arg_map.get(lhs_buf or "")
        rhs_arg = arg_map.get(rhs_buf or "")
        out_arg = arg_map.get(out_buf or "")

        if lhs_arg is None or rhs_arg is None:
            inputs = [a for a in tensor_args if a.fake_value.ndim == 2]
            if len(inputs) >= 2:
                lhs_arg = inputs[0]
                rhs_arg = inputs[1]
            if out_arg is None and len(inputs) >= 1:
                for a in tensor_args:
                    if a not in inputs[:2]:
                        out_arg = a
                        break

        assert lhs_arg is not None, "Could not find LHS tensor for matmul"
        assert rhs_arg is not None, "Could not find RHS tensor for matmul"
        assert out_arg is not None, "Could not find output tensor for matmul"

        M = env.size_hint(lhs_arg.fake_value.size(0))
        K = env.size_hint(lhs_arg.fake_value.size(1))
        N = env.size_hint(rhs_arg.fake_value.size(1))

        return lhs_arg, rhs_arg, out_arg, M, K, N

    @staticmethod
    def _emit_mpp_headers(msl_parts: list[str]) -> None:
        """Replace MSL headers with MPP-enabled includes."""
        msl_parts.clear()
        msl_parts.extend(
            [
                "#include <metal_stdlib>",
                "#include <metal_tensor>",
                "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
                "using namespace metal;",
                "using namespace mpp::tensor_ops;",
                "",
            ]
        )

    @staticmethod
    def _emit_fused_attention_body(
        msl_parts: list[str],
        body_lines: list[str],
        device_fn: DeviceFunction,
        params: list[str],
        env: object,
    ) -> None:
        """Emit fused attention MSL using MPP cooperative_tensor composition.

        Chains: matmul2d → cooperative_tensor → reduce_rows → map_iterator → store → matmul2d.

        The scores matmul outputs to a cooperative_tensor that stays in registers.
        Softmax (scale, max, exp, sum, normalize) is performed entirely on the
        cooperative_tensor using reduce_rows and map_iterator — no device memory
        round-trips until the store before the second matmul.

        Each simdgroup independently processes one TILE_M row tile. Multiple
        simdgroups in a threadgroup work on different row tiles in parallel.
        This matches the execution_simdgroup scope required by reduce_rows.

        Supports both 2D (single head) and 3D (batched B*H heads) tensors.
        For 3D, generates a 3D grid with tgid.z indexing heads.
        """
        import re

        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        assert isinstance(env, CompileEnvironment)

        # --- Phase 1: Analyze body structure ---
        all_lines: list[str] = []
        for bl in body_lines:
            for sub in bl.split("\n"):
                s = sub.strip()
                if s:
                    all_lines.append(s)

        # --- Phase 2: Extract tensor args and dimensions ---
        tensor_args = [a for a in device_fn.sorted_args() if isinstance(a, TensorArg)]
        arg_map: dict[str, TensorArg] = {}
        for arg in tensor_args:
            arg_map[arg.name] = arg

        # Trace load indirection: `auto load = q[...]` → load maps to q
        load_to_arg: dict[str, str] = {}
        for line in all_lines:
            m = re.match(r"(?:auto )?(\w+)\s*=\s*(\w+)\[", line)
            if m:
                load_to_arg[m.group(1)] = m.group(2)

        def resolve_arg(buf_name: str) -> TensorArg | None:
            if buf_name in arg_map:
                return arg_map[buf_name]
            src = load_to_arg.get(buf_name, buf_name)
            return arg_map.get(src)

        # Find matmul calls and their operands
        matmul_calls: list[tuple[str, str, str, bool]] = []
        for line in all_lines:
            # 2-arg: _metal_mm or _metal_bmm
            m = re.search(r"_metal_b?mm\((\w+),\s*(\w+)\)", line)
            if m:
                rv = "unnamed"
                m2 = re.match(r"(?:auto )?(\w+)\s*=", line)
                if m2:
                    rv = m2.group(1)
                matmul_calls.append((m.group(1), m.group(2), rv, False))
            # 3-arg: _metal_addmm or _metal_baddbmm
            for pat in [
                r"_metal_addmm\((\w+),\s*(\w+),\s*(\w+)\)",
                r"_metal_baddbmm\((\w+),\s*(\w+),\s*(\w+)\)",
            ]:
                m = re.search(pat, line)
                if m:
                    rv = "unnamed"
                    m2 = re.match(r"(?:auto )?(\w+)\s*=", line)
                    if m2:
                        rv = m2.group(1)
                    matmul_calls.append((m.group(2), m.group(3), rv, True))
                    break

        # Find the output buffer
        out_buf = None
        for line in all_lines:
            m = re.match(r"(\w+)\[", line)
            if m and m.group(1) in arg_map:
                out_buf = m.group(1)
                break

        # Resolve first matmul (scores) and last matmul (output accumulation)
        first_lhs_arg = resolve_arg(matmul_calls[0][0])
        first_rhs_arg = resolve_arg(matmul_calls[0][1])
        has_second_matmul = len(matmul_calls) >= 2
        last_rhs_arg = resolve_arg(matmul_calls[-1][1]) if has_second_matmul else None
        out_arg = arg_map.get(out_buf or "")

        # Positional fallback
        inputs_2d = [a for a in tensor_args if a.fake_value.ndim == 2]
        if first_lhs_arg is None and len(inputs_2d) >= 1:
            first_lhs_arg = inputs_2d[0]
        if first_rhs_arg is None and len(inputs_2d) >= 2:
            first_rhs_arg = inputs_2d[1]
        if last_rhs_arg is None and has_second_matmul and len(inputs_2d) >= 3:
            last_rhs_arg = inputs_2d[2]
        if out_arg is None:
            for a in tensor_args:
                if a not in (first_lhs_arg, first_rhs_arg, last_rhs_arg):
                    out_arg = a
                    break

        assert first_lhs_arg is not None
        assert first_rhs_arg is not None
        assert out_arg is not None

        # --- Phase 3: Compute dimensions ---
        is_batched = first_lhs_arg.fake_value.ndim >= 3
        if is_batched:
            batch_val = env.size_hint(first_lhs_arg.fake_value.size(0))
            M_val = env.size_hint(first_lhs_arg.fake_value.size(-2))
            K1_val = env.size_hint(first_lhs_arg.fake_value.size(-1))
            rhs0_d0 = env.size_hint(first_rhs_arg.fake_value.size(-2))
            rhs0_d1 = env.size_hint(first_rhs_arg.fake_value.size(-1))
            out_d1 = env.size_hint(out_arg.fake_value.size(-1))
        else:
            batch_val = 1
            M_val = env.size_hint(first_lhs_arg.fake_value.size(0))
            K1_val = env.size_hint(first_lhs_arg.fake_value.size(1))
            rhs0_d0 = env.size_hint(first_rhs_arg.fake_value.size(0))
            rhs0_d1 = env.size_hint(first_rhs_arg.fake_value.size(1))
            out_d1 = env.size_hint(out_arg.fake_value.size(1))
        if rhs0_d0 == K1_val:
            N_val = rhs0_d1
            transpose_rhs = False
        else:
            N_val = rhs0_d0
            transpose_rhs = True

        metal_dtype = DTYPE_TO_METAL.get(first_lhs_arg.fake_value.dtype, "float")
        config = device_fn.config
        tile_m_idx = 1 if is_batched and len(config.block_sizes) > 1 else 0
        TILE_M = (
            config.block_sizes[tile_m_idx]
            if len(config.block_sizes) > tile_m_idx
            else 64
        )
        lhs_name = first_lhs_arg.name
        rhs_name = first_rhs_arg.name
        transpose_str = "true" if transpose_rhs else "false"
        NUM_SG = config.num_warps if config.num_warps is not None else 4

        # --- Phase 4: Emit MSL with MPP cooperative_tensor composition ---
        MetalBackend._emit_mpp_headers(msl_parts)
        msl_parts.extend(
            [
                f"constant int _M = {M_val};",
                f"constant int _N = {N_val};",
                f"constant int _K = {K1_val};",
                f"constant int _TILE_M = {TILE_M};",
                f"constant int _OUT_D = {out_d1};",
            ]
        )
        if is_batched:
            msl_parts.append(f"constant int _BATCH = {batch_val};")
        msl_parts.append("")

        # MPP reduce_rows gives incorrect results for N>128 columns
        _MPP_REDUCE_ROWS_MAX_COLS = 128

        # Kernel params
        params.clear()
        buf_idx = 0
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {dt}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
        if has_second_matmul:
            params.append(f"device {metal_dtype}* _scratch [[buffer({buf_idx})]]")
        tgid_type = "uint3" if is_batched else "uint2"
        fused_params = [
            f"{tgid_type} tgid [[threadgroup_position_in_grid]]",
            "uint _sg_idx [[simdgroup_index_in_threadgroup]]",
        ]
        if N_val > _MPP_REDUCE_ROWS_MAX_COLS:
            fused_params.append("uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]")
        params.extend(fused_params)

        sig = ", ".join(params)
        msl_parts.extend([f"kernel void {device_fn.name}({sig}) {{", ""])

        # Each simdgroup independently processes one TILE_M row tile.
        # tgid.y selects the threadgroup; _sg_idx selects the simdgroup within it.
        # _tile_row = tgid.y * NUM_SG + _sg_idx gives the global tile index.
        num_tiles = (M_val + TILE_M - 1) // TILE_M
        msl_parts.extend(
            [
                f"    uint _tile_row = tgid.y * {NUM_SG} + _sg_idx;",
                f"    if (_tile_row >= {num_tiles}u) return;",
                "",
            ]
        )

        if is_batched:
            msl_parts.extend(
                ["    // Per-head pointer offsets", "    uint _head = tgid.z;"]
            )
            for arg in tensor_args:
                dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                stride = 1
                for d in range(arg.fake_value.ndim - 1, 0, -1):
                    stride *= env.size_hint(arg.fake_value.size(d))
                msl_parts.append(
                    f"    device {dt}* _h_{arg.name} = {arg.name} + _head * {stride};"
                )
            if has_second_matmul:
                scratch_stride = M_val * N_val
                msl_parts.append(
                    f"    device {metal_dtype}* _h_scratch = _scratch + _head * {scratch_stride};"
                )
            msl_parts.append("")

        # Wrap per-head 2D slices as tensor_inline
        msl_parts.append("    // Wrap buffers as tensor_inline 2D tensors")
        for arg in tensor_args:
            dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
            d0 = env.size_hint(arg.fake_value.size(-2))
            d1 = env.size_hint(arg.fake_value.size(-1))
            ptr = f"_h_{arg.name}" if is_batched else arg.name
            msl_parts.append(
                f"    auto _t_{arg.name} = tensor<device {dt}, "
                f"dextents<int32_t, 2>, tensor_inline>("
                f"\n        {ptr}, dextents<int32_t, 2>({d1}, {d0}));"
            )
        if has_second_matmul:
            scratch_ptr = "_h_scratch" if is_batched else "_scratch"
            msl_parts.append(
                f"    auto _t_scratch = tensor<device {metal_dtype}, "
                f"dextents<int32_t, 2>, tensor_inline>("
                f"\n        {scratch_ptr}, dextents<int32_t, 2>({N_val}, {M_val}));"
            )
        msl_parts.append("")

        # Two paths based on N: reduce_rows is broken for N>128.
        # - N≤128: composable cooperative_tensor pipeline (no device memory round-trip)
        # - N>128: matmul to scratch → SIMD shuffle softmax → matmul from scratch
        use_coop_softmax = N_val <= _MPP_REDUCE_ROWS_MAX_COLS

        scratch_name = "_scratch_s"
        raw_scratch = "_h_scratch" if is_batched else "_scratch"

        if use_coop_softmax:
            # --- Composable path: matmul → cooperative_tensor → reduce_rows → map → store → matmul ---
            msl_parts.extend(
                [
                    f"    // Step 1: Scores = {lhs_name} @ {rhs_name} -> cooperative_tensor",
                    "    constexpr auto _scoreDesc = matmul2d_descriptor(",
                    "        _TILE_M, _N, dynamic_length_v<int>,",
                    f"        false, {transpose_str}, false,",
                    "        matmul2d_descriptor::mode::multiply);",
                    "    matmul2d<_scoreDesc, execution_simdgroup> _scoreOp;",
                    "",
                    f"    auto _q_slice = _t_{lhs_name}.slice(0, _tile_row * _TILE_M);",
                    f"    auto _k_slice = _t_{rhs_name}.slice(0, 0);",
                    "",
                    "    // Matmul into cooperative_tensor (stays in registers)",
                    "    auto _cScores = _scoreOp.get_destination_cooperative_tensor<",
                    "        decltype(_q_slice), decltype(_k_slice), float>();",
                    "    _scoreOp.run(_q_slice, _k_slice, _cScores);",
                    "",
                    "    // Step 2: Softmax via reduce_rows + map_iterator on cooperative_tensor",
                    "    auto _cMaxRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<",
                    "        decltype(_q_slice), decltype(_k_slice), float>();",
                    "",
                    "    float _scale = rsqrt((float)_K);",
                    "    for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++)",
                    "        *_it *= _scale;",
                    "",
                    "    auto _identity_max = metal::numeric_limits<float>::lowest();",
                    "    reduce_rows(_cScores, _cMaxRow, reduction_operation::max, _identity_max);",
                    "",
                    "    if (is_iterator_compatible(_cScores, _cMaxRow)) {",
                    "        for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {",
                    "            auto _max_it = _cMaxRow.map_iterator(_it);",
                    "            *_it = exp(*_it - *_max_it);",
                    "        }",
                    "    }",
                    "",
                    "    auto _cSumRow = _scoreOp.get_row_reduction_destination_cooperative_tensor<",
                    "        decltype(_q_slice), decltype(_k_slice), float>();",
                    "    reduce_rows(_cScores, _cSumRow, reduction_operation::sum, 0.0f);",
                    "",
                    "    if (is_iterator_compatible(_cScores, _cSumRow)) {",
                    "        for (auto _it = _cScores.begin(); _it != _cScores.end(); _it++) {",
                    "            auto _sum_it = _cSumRow.map_iterator(_it);",
                    "            *_it *= (1.0f / *_sum_it);",
                    "        }",
                    "    }",
                    "",
                ]
            )

            if has_second_matmul and last_rhs_arg is not None:
                msl_parts.extend(
                    [
                        "    // Step 3: Store weights to scratch, then output = weights @ V",
                        f"    auto {scratch_name} = _t_scratch.slice(0, _tile_row * _TILE_M);",
                        f"    _cScores.store({scratch_name});",
                        "    simdgroup_barrier(mem_flags::mem_device);",
                        "",
                        f"    // Matmul #2: scratch @ {last_rhs_arg.name} -> output",
                        "    constexpr auto _outDesc = matmul2d_descriptor(",
                        "        _TILE_M, _OUT_D, dynamic_length_v<int>,",
                        "        false, false, false,",
                        "        matmul2d_descriptor::mode::multiply);",
                        "    matmul2d<_outDesc, execution_simdgroup> _outOp;",
                        f"    auto _v_slice = _t_{last_rhs_arg.name}.slice(0, 0);",
                        f"    auto _o_slice = _t_{out_arg.name}.slice(0, _tile_row * _TILE_M);",
                        f"    _outOp.run({scratch_name}, _v_slice, _o_slice);",
                    ]
                )
        else:
            # --- Device-memory path: matmul → scratch → SIMD shuffle softmax → matmul ---
            # reduce_rows is broken for N>128, so materialize scores to scratch
            # and do softmax with SIMD shuffles within each simdgroup.
            msl_parts.extend(
                [
                    f"    // Step 1: Scores = {lhs_name} @ {rhs_name} -> scratch",
                    "    constexpr auto _mm1Desc = matmul2d_descriptor(",
                    "        _TILE_M, _N, dynamic_length_v<int>,",
                    f"        false, {transpose_str}, false,",
                    "        matmul2d_descriptor::mode::multiply);",
                    "    matmul2d<_mm1Desc, execution_simdgroup> _mm1Op;",
                    "",
                    f"    auto _mm1_A = _t_{lhs_name}.slice(0, _tile_row * _TILE_M);",
                    f"    auto _mm1_B = _t_{rhs_name}.slice(0, 0);",
                    f"    auto {scratch_name} = _t_scratch.slice(0, _tile_row * _TILE_M);",
                    f"    _mm1Op.run(_mm1_A, _mm1_B, {scratch_name});",
                    "    simdgroup_barrier(mem_flags::mem_device);",
                    "",
                    "    // Step 2: SIMD shuffle softmax on scratch (per simdgroup, 32 threads)",
                    "    uint _tid = thread_index_in_simdgroup;",
                    "    float _sm_scale = rsqrt((float)_K);",
                    "",
                    "    for (int _r = 0; _r < _TILE_M; _r++) {",
                    "        int _rb = (_tile_row * _TILE_M + _r) * _N;",
                    "",
                    "        float _lm = -INFINITY;",
                    "        for (int _c = (int)_tid; _c < _N; _c += 32) {",
                    f"            float _sv = (float){raw_scratch}[_rb + _c] * _sm_scale;",
                    f"            {raw_scratch}[_rb + _c] = _sv;",
                    "            _lm = max(_lm, _sv);",
                    "        }",
                    "        for (uint _off = 16; _off > 0; _off >>= 1)",
                    "            _lm = max(_lm, simd_shuffle_down(_lm, _off));",
                    "        float _row_max = simd_broadcast_first(_lm);",
                    "",
                    "        float _ls = 0.0f;",
                    "        for (int _c = (int)_tid; _c < _N; _c += 32) {",
                    f"            float _ev = exp((float){raw_scratch}[_rb + _c] - _row_max);",
                    f"            {raw_scratch}[_rb + _c] = _ev;",
                    "            _ls += _ev;",
                    "        }",
                    "        for (uint _off = 16; _off > 0; _off >>= 1)",
                    "            _ls += simd_shuffle_down(_ls, _off);",
                    "        float _inv_sum = 1.0f / simd_broadcast_first(_ls);",
                    "",
                    "        for (int _c = (int)_tid; _c < _N; _c += 32)",
                    f"            {raw_scratch}[_rb + _c] *= _inv_sum;",
                    "        simdgroup_barrier(mem_flags::mem_device);",
                    "    }",
                    "",
                ]
            )

            if has_second_matmul and last_rhs_arg is not None:
                msl_parts.extend(
                    [
                        f"    // Step 3: Matmul #2: scratch @ {last_rhs_arg.name} -> output",
                        "    constexpr auto _mm2Desc = matmul2d_descriptor(",
                        "        _TILE_M, _OUT_D, dynamic_length_v<int>,",
                        "        false, false, false,",
                        "        matmul2d_descriptor::mode::multiply);",
                        "    matmul2d<_mm2Desc, execution_simdgroup> _mm2Op;",
                        f"    auto _mm2_B = _t_{last_rhs_arg.name}.slice(0, 0);",
                        f"    auto _mm2_C = _t_{out_arg.name}.slice(0, _tile_row * _TILE_M);",
                        f"    _mm2Op.run({scratch_name}, _mm2_B, _mm2_C);",
                    ]
                )

    @staticmethod
    def _emit_matmul_body(
        msl_parts: list[str],
        body_lines: list[str],
        device_fn: DeviceFunction,
        params: list[str],
        env: object,
    ) -> None:
        """Emit MSL body using MPP matmul2d for matrix multiplication.

        Replaces the entire kernel body with Apple MetalPerformancePrimitives
        ``matmul2d`` using ``tensor_inline`` wrappers around raw device buffers.
        """
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        assert isinstance(env, CompileEnvironment)

        lhs_arg, rhs_arg, out_arg, M, K, N = MetalBackend._extract_matmul_args(
            body_lines, device_fn, env
        )

        metal_dtype = DTYPE_TO_METAL.get(lhs_arg.fake_value.dtype, "float")

        config = device_fn.config
        TILE_M = config.block_sizes[0] if len(config.block_sizes) > 0 else 64
        TILE_N = config.block_sizes[1] if len(config.block_sizes) > 1 else 32
        NUM_SG = config.num_warps if config.num_warps is not None else 4

        # Replace headers — need MPP includes
        msl_parts.clear()
        msl_parts.extend(
            [
                "#include <metal_stdlib>",
                "#include <metal_tensor>",
                "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>",
                "using namespace metal;",
                "using namespace mpp::tensor_ops;",
                "",
                f"constant int _M = {M};",
                f"constant int _N = {N};",
                f"constant int _K = {K};",
                f"constant int _TILE_M = {TILE_M};",
                f"constant int _TILE_N = {TILE_N};",
                f"constant int _NUM_SG = {NUM_SG};",
                "",
            ]
        )

        # Replace kernel params — need uint2 tgid instead of _gid
        params.clear()
        buf_idx = 0
        for arg in device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                dt = DTYPE_TO_METAL.get(arg.fake_value.dtype, "float")
                params.append(f"device {dt}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
        params.append("uint2 tgid [[threadgroup_position_in_grid]]")

        # Always use mode::multiply (C = A*B).  The entire K-loop body is
        # replaced by a single matmul2d call with dynamic_length_v<int> which
        # handles the full K reduction internally.  multiply_accumulate would
        # read the (uninitialized) output buffer and add to it.
        mode = "multiply"

        # Add kernel signature line (will be replaced by post-processing)
        sig = ", ".join(params)
        msl_parts.append(f"kernel void {device_fn.name}({sig}) {{")

        # Tensor extents use (cols, rows) layout per MSL convention
        msl_parts.extend(
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
                "        _TILE_M, _TILE_N, dynamic_length_v<int>,",
                f"        false, false, false, matmul2d_descriptor::mode::{mode});",
                "    matmul2d<_desc, execution_simdgroups<_NUM_SG>> _op;",
                "",
                "    auto _As = _A.slice(0, tgid.y * _TILE_M);",
                "    auto _Bs = _B.slice(tgid.x * _TILE_N, 0);",
                "    auto _Cs = _C.slice(tgid.x * _TILE_N, tgid.y * _TILE_M);",
                "    _op.run(_As, _Bs, _Cs);",
            ]
        )

    @staticmethod
    def _py_to_msl(py_line: str) -> str:
        """Best-effort transformation of a single Python statement to MSL C++.

        Handles:
        - Triton / libdevice function calls → Metal stdlib equivalents
        - ``tl.reshape(...)`` → stripped (Metal works with scalars)
        - ``static_cast<T>(tl.reshape(...))`` → proper cast
        - 2D loads ``x[row, :]`` → row pointer offset (requires _RDIM loop)
        - 2D stores ``out[row, :] = val`` → row pointer store
        - ``_metal_max/sum/min(...)`` → loop-based reduction helpers
        """
        import re

        line = py_line.strip()

        # Replace Python floor division // with C integer division /
        # Use regex to only match // as an operator (between word/paren chars)
        line = re.sub(r"(?<=[\w\)])\s*//\s*(?=[\w\(])", " / ", line)

        # Strip .to(int) / .to(dtype) casts
        line = re.sub(r"\.to\(\w+\)", "", line)

        # --- Triton / inductor function rewrites ---
        # libdevice.exp → exp (Metal stdlib)
        line = re.sub(r"libdevice\.(\w+)", r"\1", line)
        # tl.reshape(expr, shape) → expr  (drop reshape, scalar in Metal)
        line = re.sub(r"tl\.reshape\(([^,]+),\s*\[[^\]]*\]\)", r"\1", line)
        # static_cast < T > expr → static_cast<T>(expr)  (fix AST unparse spacing)
        line = re.sub(
            r"static_cast\s*<\s*(\w+)\s*>\s*(\w[\w.]*(?:\([^)]*\))?)",
            r"static_cast<\1>(\2)",
            line,
        )
        # tl.cast(expr, tl.float32) → static_cast<float>(expr)
        line = re.sub(
            r"tl\.cast\(([^,]+),\s*tl\.float32\)", r"static_cast<float>(\1)", line
        )

        # --- Row-slice indexing → _RDIM marker ---
        # 2D: ``x[idx, :]`` → ``x[_gid * _RDIM + _j]``
        # 3D: ``x[batch, idx, :]`` → ``x[_gid * _RDIM + _j]`` (batch handled by emitter)
        line = re.sub(r"(\w+)\[(\w+),\s*(\w+),\s*:\]", r"\1[_gid * _RDIM + _j]", line)
        line = re.sub(r"(\w+)\[(\w+),\s*:\]", r"\1[_gid * _RDIM + _j]", line)

        # --- Reduction helpers → loop-based MSL ---
        # _metal_max(arr, dim) → reduction over _j
        line = re.sub(
            r"_metal_max\((\w+),\s*\d+\)",
            r"_reduce_max_val",
            line,
        )
        line = re.sub(
            r"_metal_sum\((\w+),\s*\d+\)",
            r"_reduce_sum_val",
            line,
        )
        line = re.sub(
            r"_metal_min\((\w+),\s*\d+\)",
            r"_reduce_min_val",
            line,
        )

        if "=" in line and not line.startswith("if ") and not line.startswith("for "):
            match = re.match(r"^([^=!<>]+)=(?!=)(.+)$", line)
            if match:
                lhs = match.group(1).rstrip()
                rhs = match.group(2).lstrip()
                if "[" in lhs:
                    return f"{lhs} = {rhs};"
                return f"auto {lhs} = {rhs};"

        return f"{line};"

    def supports_config_key(self, key: str) -> bool:
        return key in _METAL_SUPPORTED_KEYS

    def adjust_block_size_constraints(
        self, block_specs: list[object], ndim: int
    ) -> None:
        from ..autotuner.config_spec import BlockSizeSpec

        for spec in block_specs:
            if isinstance(spec, BlockSizeSpec):
                spec.update_min(METAL_SIMD_WIDTH)
        # For matmul kernels (3+ block specs), pin the inner reduction
        # K-tile. MPP matmul2d uses dynamic_length_v<int> for K, so the
        # K-tile size has no effect. Pinning it prevents the autotuner
        # from wasting budget exploring a no-op dimension.
        if len(block_specs) >= 3:
            last = block_specs[-1]
            if isinstance(last, BlockSizeSpec):
                # Pin to the next power of 2 of the size hint
                from .._utils import next_power_of_2

                pinned = next_power_of_2(max(last.size_hint, METAL_SIMD_WIDTH))
                last.update_min(pinned)
                last.update_max(pinned)

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
        """Build Metal launcher arguments based on kernel classification.

        Uses tensor arg ndim/count as a proxy for kernel type. This
        classification must be consistent with the body-text based
        classification in ``generate_msl_function``. Both use
        ``MetalKernelKind`` to keep the logic aligned.

        Note: ``build_launcher_args`` is called BEFORE
        ``generate_msl_function``, so we can't use ``self._kernel_kind``
        directly — we must re-derive the classification from tensor args.
        """
        from .compile_environment import CompileEnvironment
        from .device_function import TensorArg

        out = [*args]

        # Classify from tensor args (proxy for body-text classification)
        tensor_nd = [
            a
            for a in (sorted_args or [])
            if isinstance(a, TensorArg) and a.fake_value.ndim >= 2
        ]
        tensor_2d = [a for a in tensor_nd if a.fake_value.ndim == 2]
        if len(tensor_nd) >= 4:
            kind = MetalKernelKind.FUSED_ATTENTION
        elif len(tensor_2d) >= 3:
            kind = MetalKernelKind.MATMUL
        elif len(tensor_nd) >= 1:
            kind = MetalKernelKind.SOFTMAX
        else:
            kind = MetalKernelKind.ELEMENTWISE

        if kind == MetalKernelKind.FUSED_ATTENTION and sorted_args is not None:
            env = CompileEnvironment.current()
            tensor_args = [
                a
                for a in sorted_args
                if isinstance(a, TensorArg) and a.fake_value.ndim >= 2
            ]
            first_arg = tensor_args[0]
            is_batched = first_arg.fake_value.ndim >= 3
            batch_val = env.size_hint(first_arg.fake_value.size(0)) if is_batched else 1
            M_val = env.size_hint(first_arg.fake_value.size(-2))
            K1_val = env.size_hint(first_arg.fake_value.size(-1))
            second_arg = tensor_args[1]
            d0 = env.size_hint(second_arg.fake_value.size(-2))
            d1 = env.size_hint(second_arg.fake_value.size(-1))
            N1_val = d1 if d0 == K1_val else d0
            tile_m_idx = 1 if is_batched and len(config.block_sizes) > 1 else 0
            TILE_M = (
                config.block_sizes[tile_m_idx]
                if len(config.block_sizes) > tile_m_idx
                else 64
            )
            num_tiles = (M_val + TILE_M - 1) // TILE_M
            scratch_size = batch_val * M_val * N1_val
            NUM_SG = config.num_warps if config.num_warps is not None else 4
            # Each threadgroup has NUM_SG simdgroups, each handling a tile.
            # tile_row = tgid.y * NUM_SG + sg_idx, so grid_y = ceil(num_tiles / NUM_SG).
            grid_y = (num_tiles + NUM_SG - 1) // NUM_SG
            tpg = METAL_SIMD_WIDTH * NUM_SG
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
            env = CompileEnvironment.current()
            tensor_2d = [
                a
                for a in sorted_args
                if isinstance(a, TensorArg) and a.fake_value.ndim == 2
            ]
            # Find A[M,K] @ B[K,N] = C[M,N] from tensor shapes
            M = N = 0
            for i in range(len(tensor_2d)):
                for j in range(len(tensor_2d)):
                    for k in range(len(tensor_2d)):
                        if len({i, j, k}) < 3:
                            continue
                        a, b, c = tensor_2d[i], tensor_2d[j], tensor_2d[k]
                        a_m = env.size_hint(a.fake_value.size(0))
                        a_k = env.size_hint(a.fake_value.size(1))
                        b_k = env.size_hint(b.fake_value.size(0))
                        b_n = env.size_hint(b.fake_value.size(1))
                        c_m = env.size_hint(c.fake_value.size(0))
                        c_n = env.size_hint(c.fake_value.size(1))
                        if a_k == b_k and a_m == c_m and b_n == c_n:
                            M, N = a_m, b_n
                            break
                    if M > 0:
                        break
                if M > 0:
                    break
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
            block_size = config.block_sizes[0] if config.block_sizes else 256
            out.append(f"_block_size={block_size}")
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
