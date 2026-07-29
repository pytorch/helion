"""MetalBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

import torch

from ... import exc
from ..backend import Backend
from ..backend import _largest_divisor_at_most
from ..cute.backend import CuteBackend

if TYPE_CHECKING:
    from torch._inductor.ops_handler import OpsHandler

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..device_function import Argument
    from ..device_function import DeviceFunction
    from ..tile_strategy import TileStrategy

    InductorOpOverrides = OpsHandler[Any]


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
        from .metal_overrides import MetalOverrides

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
        from ..device_function import SymbolArgument

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
        from ..device_function import DeviceFunction

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

        from ..host_function import HostFunction

        device_ir = HostFunction.current().device_ir
        # Only adjust the loop strategy for the root grid.  MPPGraphInfo emits
        # the cooperative K-loop internally; nested/device loops should keep
        # their normal Metal/CuTe strategy.
        if not device_ir.grid_block_ids or block_ids != device_ir.grid_block_ids[0]:
            return config
        if len(block_ids) < 2:
            return config

        from ...runtime.config import Config
        from ..compile_environment import CompileEnvironment
        from ..cute.thread_budget import MAX_THREADS_PER_BLOCK

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
