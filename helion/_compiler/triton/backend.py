"""TritonBackend / TileIRBackend classes, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

import base64
import contextlib
import functools
import math
import os
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import torch

from ... import exc
from ..backend import Backend
from ..backend import log

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_fragment import ConfigSpecFragment
    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..device_function import Argument

    InductorOpOverrides = OpsHandler[Any]


@functools.cache
def _triton_jit_supports_do_not_specialize() -> bool:
    try:
        import inspect

        import triton
    except ImportError:
        return False

    params = inspect.signature(triton.jit).parameters
    return "do_not_specialize" in params and "do_not_specialize_on_alignment" in params


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
        from ..device_function import TensorArg

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
            from ..._compat import is_hip

            return is_hip()
        if key == "matrix_instr_nonkdim":
            from ..._compat import supports_amd_cdna_tunables

            return supports_amd_cdna_tunables()
        if key == "xcd_remap":
            from ..._compat import supports_amd_cdna_tunables

            # Accepted on all AMD CDNA.  On single-XCD devices it normalizes to a
            # no-op (ConfigSpec.normalize) and is excluded from the search space
            # (ConfigSpec.flat_config) rather than rejected.
            return supports_amd_cdna_tunables()

        from ..._compat import get_mtia_tunable_fragments
        from ..._compat import supports_mtia_tunables

        if key in get_mtia_tunable_fragments():
            return supports_mtia_tunables()
        return super().supports_config_key(key)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        from ..._compat import get_mtia_tunable_fragments
        from ..._compat import is_hip
        from ..._compat import supports_amd_cdna_tunables
        from ..._compat import supports_mtia_tunables
        from ...autotuner.config_fragment import EnumFragment

        if not is_hip() and not supports_mtia_tunables():
            return {}
        fragments: dict[str, ConfigSpecFragment] = {}
        if is_hip():
            # A value of 0 leaves occupancy unconstrained, matching Triton's default.
            fragments["waves_per_eu"] = EnumFragment(choices=(0, 1, 2, 3, 4))
            if supports_amd_cdna_tunables():
                fragments["matrix_instr_nonkdim"] = EnumFragment(choices=(0, 16, 32))

        if supports_mtia_tunables():
            fragments.update(get_mtia_tunable_fragments())

        return fragments

    def setup_compile_cache_dir(self, device_index: int) -> None:
        if "TRITON_CACHE_DIR" not in os.environ:
            from ...autotuner.local_cache import helion_triton_cache_dir

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
        from ...runtime.config import Config

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
        from ...runtime.config import Config

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
        from ..compile_environment import CompileEnvironment
        from ..device_function import SymbolArgument
        from ..device_function import TensorSizeArg
        from ..device_function import TensorStrideArg

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
        from .overrides import HelionTritonOverrides

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
        from ..._compat import supports_maxnreg

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

        from ...autotuner.config_spec import _get_backend_tunable_keys

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
        from ...autotuner.config_fragment import PowerOfTwoFragment

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
