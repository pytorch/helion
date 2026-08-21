"""FlyDSLBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..backend import Backend

if TYPE_CHECKING:
    import ast

    from torch._inductor.ops_handler import OpsHandler

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

    InductorOpOverrides = OpsHandler[Any]


class FlyDSLBackend(Backend):
    """FlyDSL (ROCm) code generation backend."""

    _DTYPE_MAP: ClassVar[dict[torch.dtype, str]] = {
        torch.float16: "fx.Float16",
        torch.bfloat16: "fx.BFloat16",
        torch.float32: "fx.Float32",
        torch.float64: "fx.Float64",
        torch.int32: "fx.Int32",
        torch.int64: "fx.Int64",
        torch.bool: "fx.Bool",
    }

    _ACC_TYPE: ClassVar[dict[torch.dtype, str]] = {
        torch.float16: "fx.Float32",
        torch.bfloat16: "fx.Float32",
        torch.float32: "fx.Float32",
        torch.float64: "fx.Float64",
        torch.int32: "fx.Int32",
        torch.int64: "fx.Int64",
        torch.bool: "fx.Int32",
    }

    _SUPPORTED_CONFIG_KEYS: frozenset[str] = frozenset(
        {
            "block_sizes",
            "num_warps",
            "num_threads",
            "reduction_loops",
        }
    )

    @property
    def name(self) -> str:
        return "flydsl"

    @property
    def experimental(self) -> bool:
        return True

    def validate_environment(self) -> None:
        try:
            import flydsl  # noqa: F401  # pyrefly: ignore[missing-import]
        except ImportError as e:
            raise exc.BackendUnsupported(
                self.name,
                "flydsl is not installed; install it with: pip install flydsl",
            ) from e

    def dtype_str(self, dtype: torch.dtype) -> str:
        if dtype not in self._DTYPE_MAP:
            raise exc.BackendUnsupported(self.name, f"dtype: {dtype}")
        return self._DTYPE_MAP[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype not in self._ACC_TYPE:
            raise exc.BackendUnsupported(self.name, f"acc_type for: {dtype}")
        return self._ACC_TYPE[dtype]

    @property
    def function_decorator(self) -> str:
        # Pin known_block_size = 64*bm (bm warps, one warp per row). This is a
        # codegen-time constant from the config and matches _num_threads passed
        # by launcher_keyword_args exactly, so flydsl's block-vs-known check
        # passes. Required for blocks > 256 threads (bm > 4), which exceed the
        # AMDGPU default max_flat_workgroup_size.
        n_threads = getattr(self, "_flydsl_num_threads", 64)
        return f"flyc.kernel(known_block_size=[{n_threads}, 1, 1])"

    @property
    def constexpr_type(self) -> str:
        return "fx.Constexpr"

    @property
    def default_launcher_name(self) -> str:
        return "_default_flydsl_launcher"

    def max_reduction_threads(self) -> int | None:
        # 64 = one wavefront: one warp per row. A finite value routes ``:``
        # (full-column) mappings through the looped strategy with a 64-lane
        # thread count, which the elementwise load/store codegen depends on.
        return 64

    def max_reduction_loop(self) -> int | None:
        return 64

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "torch": "import torch",
            "flyc": "import flydsl.compiler as flyc",
            "fx": "import flydsl.expr as fx",
            "fmath": "from flydsl.expr import math as fmath",
            "arith": "from flydsl.expr import arith",
            "rocdl": "from flydsl.expr import rocdl",
            "gpu": "from flydsl.expr import gpu",
            "full": "from flydsl.expr.vector import full",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "_default_flydsl_launcher": (
                "from helion.runtime import default_flydsl_launcher"
                " as _default_flydsl_launcher"
            ),
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"fx.block_idx.{'xyz'[dim]}"

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        # Elementwise: block = 64*bm (bm warps, one warp per row).
        # AMD caps a workgroup at 1024 threads.
        bs = config.block_sizes
        bm = int(bs[0])
        n_threads = 64 * bm
        if n_threads > 1024:
            raise exc.BackendUnsupported(
                self.name, f"block too large: {n_threads} threads"
            )
        return [f"_num_threads={n_threads}"]

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{expr_str}.to({dtype_str})"

    def cast_scalar_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        # Bare Python scalars (e.g. a mean divisor lifted from index_expr) have
        # no ``.to``; use the fx dtype constructor, which accepts int/float.
        # Not valid for Vector casts -- only index_expr scalars route here.
        return expr_from_string(f"{self.dtype_str(target_dtype)}({{x}})", x=x)

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    def supports_config_key(self, key: str) -> bool:
        return key in self._SUPPORTED_CONFIG_KEYS

    def supports_precompile(self) -> bool:
        return False

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        # Warp-per-row model: row (first) tile -> warps, column (last) tile ->
        # lanes. Cap bm so 64*bm <= 1024 (AMD max workgroup) => bm <= 16. Pin the
        # column block to 256 (one warp-pass = 64 lanes x vec_width 4); bn>256
        # would silently drop columns and bn<256 would underfill the warp.
        from ...autotuner.config_spec import BlockSizeSpec

        specs = [s for s in block_specs if isinstance(s, BlockSizeSpec)]
        if not specs:
            return
        specs[0].update_max(16)
        if ndim >= 2:
            # Pin ALL column dims (indices 1..ndim-1) to [256, 2048].
            # For kernels with multiple inner hl.tile(n) loops (e.g. softmax_bwd
            # with 3 dims: [bm, n1, n2]) the default spec gives non-column dims a
            # small default (e.g. 16), which OOBs in the lane-index formula.
            for col in specs[1:]:
                col.update_min(256)
                col.update_max(
                    2048
                )  # allow bn = W*256 for W in {1,2,4,8}; autotune restricts

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        # Elementwise-only search space: the sole free knob is bm (rows per
        # block = warps); every column dim is pinned to 256 by
        # adjust_block_size_constraints, and bm is capped at 16 (64*bm <= 1024
        # AMD max). FlyDSL has no precompile and its JIT does not survive the
        # subprocess benchmark workers the generic search spawns, so enumerate
        # the few valid configs and FiniteSearch them in-process.
        from ...runtime.config import Config

        spec = bound_kernel.config_spec
        default = spec.default_config()
        default_bs = default.config.get("block_sizes")
        if not isinstance(default_bs, list) or not default_bs:
            return default
        block_sizes = [int(b) for b in default_bs]

        row_hint = spec.block_sizes[0].size_hint if spec.block_sizes else 1
        candidates: list[Config] = []
        seen: set[tuple[int, ...]] = set()
        for bm in (1, 2, 4, 8, 16):
            if bm > max(row_hint, 1):
                continue
            bs = list(block_sizes)
            bs[0] = bm
            key = tuple(bs)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Config(block_sizes=list(bs)))

        if not candidates:
            return default
        if len(candidates) == 1:
            return candidates[0]

        # In-process benchmarking: FlyDSL's JIT/HIP-stream state does not
        # survive the precompile/benchmark subprocess workers (the fork/spawn
        # worker pipe closes before returning a result), so disable both the
        # precompile and the benchmark subprocess paths.
        bound_kernel.settings.autotune_precompile = None
        bound_kernel.settings.autotune_benchmark_subprocess = False

        from ...autotuner import FiniteSearch

        return FiniteSearch(bound_kernel, args, candidates).autotune()

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from ...language import memory_ops

        _BITS: dict[torch.dtype, int] = {
            torch.float16: 16,
            torch.bfloat16: 16,
            torch.float32: 32,
            torch.float64: 64,
            torch.int32: 32,
            torch.int64: 64,
        }

        # Reset per-compilation state so helpers are re-emitted on each compile.
        self._flydsl_helpers_emitted = False
        # Elementwise regime: block_sizes = [bm] (or [bm, 256]) -> bm rows/block,
        # one warp (64 lanes) per row, block = 64*bm threads. W is always 1.
        bs = getattr(config, "block_sizes", None) or [1]
        bm = int(bs[0]) if bs else 1

        self._flydsl_warps_per_row = 1
        self._flydsl_bm = bm
        self._flydsl_num_threads = 64 * bm

        self._tensor_use_buffer: dict[int, bool] = {}
        self._tensor_vec_width: dict[int, int] = {}

        for graph_info in graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target not in (memory_ops.load, memory_ops.store):
                    continue

                tensor_node = node.args[0]
                if not isinstance(tensor_node, torch.fx.Node):
                    continue
                tensor = tensor_node.meta.get("val")
                if not isinstance(tensor, torch.Tensor):
                    continue

                tid = id(tensor)
                self._tensor_use_buffer[tid] = True  # always vectorized buffer path
                bits = _BITS.get(tensor.dtype, 32)
                self._tensor_vec_width[tid] = 128 // bits

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        # Row (grid) tile -> warps. A block holds bm rows as bm warps; warp
        # w = thread_idx.x // 64 owns row (block_idx.x * bm + w). Force dim x
        # (flat block) regardless of the axis Helion assigns.
        # W>1 regime is one row per block (bm=1) -> row = block_idx.x, no warp offset.
        if getattr(self, "_flydsl_warps_per_row", 1) > 1:
            return offset_var
        if block_size_var == "1":
            return offset_var
        return f"({offset_var}) + fx.thread_idx.x // 64"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        # Column (loop) tile -> lanes. chunk = col_offset//vec + lane_id.
        # W=1: lane_id = thread_idx.x % 64 (one warp/row).
        # W>1: W*64 lanes cover one row -> lane_id = full thread_idx.x (no % 64).
        if block_size_var == "1":
            return offset_var
        if getattr(self, "_flydsl_warps_per_row", 1) > 1:
            return f"({offset_var}) // 4 + fx.thread_idx.x"
        return f"({offset_var}) // 4 + fx.thread_idx.x % 64"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        # Column lane chunk: element_offset//vec + lane_id.
        # W>1 uses the full thread id (W*64 lanes/row); W=1 uses thread_idx.x % 64.
        if getattr(self, "_flydsl_warps_per_row", 1) > 1:
            return f"{offsets_var} = ({lid}) // 4 + fx.thread_idx.x"
        return f"{offsets_var} = ({lid}) // 4 + fx.thread_idx.x % 64"

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        # Lane mask (flat block, dim x). W>1 spans W*64 lanes/row -> full thread id.
        if getattr(self, "_flydsl_warps_per_row", 1) > 1:
            return f"fx.thread_idx.x < ({block_size_var})"
        return f"fx.thread_idx.x % 64 < ({block_size_var})"

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        dim = "xyz"[axis]
        return f"fx.thread_idx.{dim} * {elements_per_thread} + fx.Int32({offset_var})"

    def lane_offset_expr(self, lane_var: str) -> str:
        return f"fx.Int32({lane_var})"

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            return f"{tensor_name}[0]"
        return f"{tensor_name}[{index_expr}]"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        # Lane index for the column (``:``) mapping: one warp per row -> lane =
        # thread_idx.x % 64.
        return "fx.thread_idx.x % 64"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "fx.Int32(0)"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.triton import TritonOverrides

        backend_name = self.name
        fly_dtype_str = self.dtype_str

        class FlyDSLOpOverrides(TritonOverrides):
            @staticmethod
            def constant(value: object, dtype: torch.dtype) -> str:
                import math as _math

                if isinstance(value, float) and _math.isinf(value):
                    v = "float('inf')" if value > 0 else "float('-inf')"
                elif isinstance(value, float) and _math.isnan(value):
                    v = "float('nan')"
                else:
                    v = repr(value)
                # flydsl scalar constants use the fx dtype constructor, not
                # Triton's tl.full(...).
                return f"{fly_dtype_str(dtype)}({v})"

            @staticmethod
            def exp(x: str) -> str:
                return f"fmath.exp({x})"

            @staticmethod
            def exp2(x: str) -> str:
                return f"fmath.exp2({x})"

            @staticmethod
            def expm1(x: str) -> str:
                return f"fmath.expm1({x})"

            @staticmethod
            def log(x: str) -> str:
                return f"fmath.log({x})"

            @staticmethod
            def log2(x: str) -> str:
                return f"fmath.log2({x})"

            @staticmethod
            def log10(x: str) -> str:
                return f"fmath.log10({x})"

            @staticmethod
            def log1p(x: str) -> str:
                return f"fmath.log1p({x})"

            @staticmethod
            def sqrt(x: str) -> str:
                return f"fmath.sqrt({x})"

            @staticmethod
            def rsqrt(x: str) -> str:
                return f"fmath.rsqrt({x})"

            @staticmethod
            def cbrt(x: str) -> str:
                return f"fmath.cbrt({x})"

            @staticmethod
            def abs(x: str) -> str:
                return f"fmath.absf({x})"

            @staticmethod
            def sin(x: str) -> str:
                return f"fmath.sin({x})"

            @staticmethod
            def cos(x: str) -> str:
                return f"fmath.cos({x})"

            @staticmethod
            def tan(x: str) -> str:
                return f"fmath.tan({x})"

            @staticmethod
            def asin(x: str) -> str:
                return f"fmath.asin({x})"

            @staticmethod
            def acos(x: str) -> str:
                return f"fmath.acos({x})"

            @staticmethod
            def atan(x: str) -> str:
                return f"fmath.atan({x})"

            @staticmethod
            def atan2(x: str, y: str) -> str:
                return f"fmath.atan2({x}, {y})"

            @staticmethod
            def sinh(x: str) -> str:
                return f"fmath.sinh({x})"

            @staticmethod
            def cosh(x: str) -> str:
                return f"fmath.cosh({x})"

            @staticmethod
            def tanh(x: str) -> str:
                return f"fmath.tanh({x})"

            @staticmethod
            def asinh(x: str) -> str:
                return f"fmath.asinh({x})"

            @staticmethod
            def acosh(x: str) -> str:
                return f"fmath.acosh({x})"

            @staticmethod
            def atanh(x: str) -> str:
                return f"fmath.atanh({x})"

            @staticmethod
            def erf(x: str) -> str:
                return f"fmath.erf({x})"

            @staticmethod
            def erfc(x: str) -> str:
                return f"fmath.erfc({x})"

            @staticmethod
            def sigmoid(x: str) -> str:
                # fmath has no sigmoid; express it via exp.
                return f"(1.0 / (1.0 + fmath.exp(-({x}))))"

            @staticmethod
            def floor(x: str) -> str:
                return f"fmath.floor({x})"

            @staticmethod
            def ceil(x: str) -> str:
                return f"fmath.ceil({x})"

            @staticmethod
            def trunc(x: str) -> str:
                return f"fmath.trunc({x})"

            @staticmethod
            def round(x: str) -> str:
                return f"fmath.round({x})"

            @staticmethod
            def copysign(x: str, y: str) -> str:
                return f"fmath.copysign({x}, {y})"

            @staticmethod
            def isnan(x: str) -> str:
                return f"fmath.isnan({x})"

            @staticmethod
            def isinf(x: str) -> str:
                return f"fmath.isinf({x})"

            @staticmethod
            def maximum(a: str, b: str) -> str:
                return f"({a}).maximumf({b})"

            @staticmethod
            def minimum(a: str, b: str) -> str:
                # flydsl Vector has no ``.minimumf``; min(a,b) = -max(-a,-b).
                return f"(-((-({a})).maximumf((-({b})))))"

            @staticmethod
            def where(a: str, b: str, c: str) -> str:
                return f"({a}).select({b}, {c})"

            def __getattr__(self, name: str) -> object:
                # Guard against silent Triton fallthrough: any op not
                # explicitly overridden here (and not defined on
                # TritonOverrides) would otherwise emit Triton-syntax such
                # as tl.libdevice.* into a FlyDSL kernel and fail at trace
                # time. Fail loudly with a clear message instead.
                if name.startswith("_"):
                    raise AttributeError(name)
                raise exc.BackendUnsupported(backend_name, f"op {name!r}")

        return FlyDSLOpOverrides()

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"{dtype}(0)"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        dtype_str = self.dtype_str(dtype)
        return f"{dtype_str}({value_expr})"

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"({mask}).select({true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        # flydsl Vector has no ``.minimumf``; min(a,b) = -max(-a,-b).
        return f"(-((-({a})).maximumf((-({b})))))"
