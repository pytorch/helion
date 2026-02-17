from __future__ import annotations

import abc
import functools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from .. import exc
from .ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    import torch
    from torch._inductor.ops_handler import OpsHandler

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

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        """Generate a backend-specific range expression, or None to use the default."""
        del begin, end, step
        return None

    def arange_expr(
        self, offsets_var: str, lid: str, block_size_var: str, dtype: str
    ) -> str:
        """Generate a backend-specific arange expression for loop offsets."""
        return f"{offsets_var} = {lid} * {block_size_var} + tl.arange(0, {block_size_var}).to({dtype})"

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

    @property
    @abc.abstractmethod
    def default_launcher_name(self) -> str:
        """Name of the default host-side launcher symbol for this backend."""
        ...

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
        del arg, tensor_host_args
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        """Generate preamble statements for scalar arguments in the device function.

        Backends can override to dereference scalar refs, etc.
        """
        del arg
        return []

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
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
        raise exc.BackendUnsupported(self.name, "autotune")


class TritonBackend(Backend):
    """Triton code generation backend."""

    @property
    def name(self) -> str:
        return "triton"

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

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(f"tl.cast({{x}}, {self.dtype_str(target_dtype)})", x=x)

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

        for key in ("waves_per_eu", "matrix_instr_nonkdim", "num_ctas", "occupancy"):
            if key in config:
                args.append(f"{key}={config[key]}")

        if "maxnreg" in config and config["maxnreg"] is not None and supports_maxnreg():
            args.append(f"maxnreg={config['maxnreg']}")

        return args

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
    ) -> list[str]:
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        force = force or bound_kernel.settings.force_autotune
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
    """Pallas (JAX) code generation backend."""

    @property
    def name(self) -> str:
        return "pallas"

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_JAX_DTYPE:
            raise ValueError(f"Unsupported dtype for Pallas backend: {dtype}")
        return _TORCH_TO_JAX_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        import torch as _torch

        # Promote half-precision types to float32 for numerical stability
        if dtype in (_torch.float16, _torch.bfloat16):
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
            "_default_pallas_launcher": "from helion.runtime import default_pallas_launcher as _default_pallas_launcher",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        del index_dtype
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
        self, offsets_var: str, lid: str, block_size_var: str, dtype: str
    ) -> str:
        return f"{offsets_var} = {lid} * {block_size_var} + jnp.arange(0, {block_size_var}, dtype={dtype})"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.pallas import PallasKernelOverrides

        return PallasKernelOverrides()

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            f"lax.convert_element_type({{x}}, {self.dtype_str(target_dtype)})", x=x
        )

    # TODO(oulgen): once https://github.com/jax-ml/jax/pull/35116 is merged
    # and released, swap to static_argnums API instead of converting scalars
    # to 0-dim tensors.
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
                f"{tensor_host_args[0]}.device" if tensor_host_args else "'cuda'"
            )
            return f"torch.tensor({host_str}, device={device_expr})"
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        from .ast_extension import statement_from_string
        from .device_function import SymbolArgument
        from .device_function import TensorSizeArg
        from .device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            return [statement_from_string(f"{arg.name} = {arg.name}[...]")]
        return []

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

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
        import torch as _torch

        if dtype in (_torch.float16, _torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        # CuTe shape/type constants are represented as regular Python values.
        return "int"

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
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"cute.arch.block_idx()[{dim}]"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        return CuteDSLOpOverrides()

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(f"{self.dtype_str(target_dtype)}({{x}})", x=x)

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        return ["block=(1, 1, 1)"]

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
    ) -> list[str]:
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        if not tensor_host_args:
            raise exc.BackendUnsupported(self.name, "kernel launch without tensor args")
        return [*args, *self.launcher_keyword_args(config, has_barrier=has_barrier)]

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from .tile_strategy import CutePointwiseTileStrategy

        return CutePointwiseTileStrategy(fn, block_ids)

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        return bound_kernel.config_spec.default_config()
