"""FlyDSLBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..backend import Backend

if TYPE_CHECKING:
    import ast

    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..compile_environment import CompileEnvironment
    from ..device_function import Argument
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

    InductorOpOverrides = OpsHandler[Any]


def _flydsl_minimum_expr(a: str, b: str) -> str:
    # flydsl Vector has no ``.minimumf``; min(a,b) = -max(-a,-b).
    return f"(-((-({a})).maximumf((-({b})))))"


def _has_user_tiled_reduction(env: CompileEnvironment) -> bool:
    """Whether the kernel has an explicit ``hl.tile(n)`` reduction.

    True when a reduction block_id lives in ``block_sizes`` (not
    ``reduction_loops``). For these kernels every non-row block dim is a column
    tile in the warp-per-row model (indices = offset//4 + lane), so it must be a
    multiple of 256.
    """
    bs_ids = env.config_spec.block_sizes.valid_block_ids()
    return any(info.reduction and info.block_id in bs_ids for info in env.block_sizes)


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
            # Shared per-reduction-block vector width V (elems/thread). The
            # looped reduction derives thread_count = chunk // V from it; V also
            # selects the load width (V=4 -> 128-bit, V=8 -> 128-bit fp16).
            "cute_vector_widths",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        # Set per-compile by pre_codegen; initialized here so function_decorator
        # and the memory-op codegen can read them directly without defaults.
        self._flydsl_num_threads: int = 64
        # Wavefronts cooperating on one row: 1 = warp-shuffle only (W=1), >1 =
        # cross-wave smem block reduce. Set per-compile by pre_codegen.
        self._flydsl_warps_per_row: int = 1
        self._tensor_use_buffer: dict[int, bool] = {}
        # Both reset each compile by pre_codegen. ``_needs_warp_helpers`` is set
        # by reduction_expr when a reduction is generated, so the warp-reduce
        # helper defs are emitted (from scalar_arg_preamble, which runs after the
        # body) only for kernels that actually reduce -- not every kernel.
        # ``_helpers_emitted`` then guards against emitting them more than once.
        self._flydsl_needs_warp_helpers: bool = False
        self._flydsl_helpers_emitted: bool = False

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
        n_threads = self._flydsl_num_threads
        return f"flyc.kernel(known_block_size=[{n_threads}, 1, 1])"

    @property
    def constexpr_type(self) -> str:
        return "fx.Constexpr"

    @property
    def default_launcher_name(self) -> str:
        return "_default_flydsl_launcher"

    def max_reduction_threads(self) -> int | None:
        return 64

    def max_reduction_loop(self) -> int | None:
        # chunk = 64 * W * V (W wavefronts x 64 lanes x V contiguous elems each).
        # The autotuner enumerates chunks up to W=16 (thread_count 1024) x V.
        return 8192

    @staticmethod
    def _flydsl_looped_thread_count(config: Config, bm: int) -> int | None:
        """thread_count (= 64*W) for the whole-row looped reduction, else None.

        W>1 (cross-wave smem fold via ``_flydsl_bsum``) requires one row per block
        (bm==1). thread_count = chunk // V, clamped to [64, 1024] and rounded to
        whole warps; V comes from the shared ``cute_vector_widths`` knob. Kept in
        sync with the ``looped_reduction_thread_count`` override (which the
        reduction strategy calls) so the launcher block dim matches codegen.
        """
        if bm != 1:
            return None
        rl = config.reduction_loops
        if not rl or rl[0] is None:
            return None
        vw = cast("list[int]", config.config.get("cute_vector_widths", []) or [])
        # flydsl loads are >=128-bit (V>=4); matches the strategy's clamp so the
        # default (unset) preserves the 1-warp behavior at chunk=256.
        v = max(4, int(vw[0]) if vw else 1)
        chunk = int(rl[0])
        return max(64, min(1024, (chunk // v // 64) * 64))

    def wrap_reduction_accumulator(
        self,
        acc_full: str,
        *,
        thread_count: int,
        loop_block_size: int,
        acc_dtype: torch.dtype,
    ) -> str:
        # flydsl's runtime scf.for carries the accumulator as an iter_arg whose
        # init type must match the vector<Vxf32> the loop body yields. V =
        # per-lane elements = loop chunk / thread count (identical to the
        # redcol_vec derived in memory_ops). thread_count = 64*W here, so
        # loop_block_size // thread_count = V regardless of W.
        # Only reached on the looped path, where thread_count is always > 0; a
        # hard failure here is clearer than seeding V=1 and hitting a confusing
        # downstream vector-type mismatch.
        assert thread_count > 0, "flydsl looped reduction needs thread_count > 0"
        vec = max(1, loop_block_size // thread_count)
        return f"fx.Vector.filled({vec}, {acc_full}, {self.dtype_str(acc_dtype)})"

    def looped_reduction_thread_count(
        self,
        *,
        requested: int,
        block_size: int,
        block_index: int,
        config: Config,
        config_spec: ConfigSpec,
    ) -> int | None:
        # flydsl pins one wavefront (64 threads) per row. For bm>1 that is
        # 64*bm threads = bm warps, each warp folding its own row -> per-row
        # thread_count is 64. For bm==1 the row may span W wavefronts:
        # thread_count = chunk // V (V from the shared cute_vector_widths knob),
        # so W = thread_count // 64 and V are independent (chunk = 64*W*V);
        # memory_ops derives redcol_vec = chunk // thread_count = V for free.
        # (max_reduction_threads=64 caps ``requested`` at 64, so W>1 only comes
        # from this override recomputing the count from the chunk.)
        _bm = int(config.block_sizes[0]) if config.block_sizes else 1
        if _bm != 1:
            return 64
        _vw = cast("list[int]", config.config.get("cute_vector_widths", []) or [])
        # flydsl loads are >=128-bit (V>=4); V=1/2 would waste bandwidth, so
        # clamp up. Default (unset -> 1) becomes V=4 = the 1-warp behavior at
        # chunk=256; W>1 comes from larger chunks (chunk = 64*W*V).
        _v = max(4, config_spec.cute_vector_widths.config_get(_vw, block_index, 1) or 1)
        _tc = (block_size // _v // 64) * 64
        return max(64, min(1024, _tc))

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
            "ReductionOp": "from flydsl.expr.vector import ReductionOp",
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
        # W=1: block = 64*bm (bm warps, one warp per row). bn is pinned to 256.
        # AMD caps a workgroup at 1024 threads.
        bs = config.block_sizes
        bm = int(bs[0])
        n_threads = 64 * bm
        # Whole-row looped reduction: threads = thread_count (W=1 -> 64).
        _tc = self._flydsl_looped_thread_count(config, bm)
        if _tc is not None:
            n_threads = _tc
        if n_threads > 1024:
            raise exc.BackendUnsupported(
                self.name, f"block too large: {n_threads} threads"
            )
        return [f"_num_threads={n_threads}"]

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{expr_str}.to({dtype_str})"

    def cast_scalar_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        # A bare scalar has no ``.to``; use the fx dtype constructor instead.
        # Only index_expr scalars route here, never Vector casts.
        return expr_from_string(f"{self.dtype_str(target_dtype)}({{x}})", x=x)

    def expands_broadcast_dims(self) -> bool:
        # FlyDSL per-thread vectors carry the tile/row axis implicitly, so a
        # Triton-style [None, :] broadcast-expand is both unsupported and
        # unnecessary -- skip it.
        return False

    def reduction_block_size_is_inlined_constexpr(self) -> bool:
        return True

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
        # Warp-per-row: row tile -> warps, column tile -> lanes. Cap bm at 16
        # (64*bm <= 1024 AMD max). Pin the column block to 256 (64 lanes x
        # vec_width 4); bn>256 drops columns, bn<256 underfills the warp.
        from ...autotuner.config_spec import BlockSizeSpec

        specs = [s for s in block_specs if isinstance(s, BlockSizeSpec)]
        if not specs:
            return
        specs[0].update_max(16)
        if ndim >= 2:
            # Pin ALL column dims (1..ndim-1) to [256, 2048]. Kernels with
            # multiple inner hl.tile(n) loops otherwise default some dims to 16,
            # which OOBs in the lane-index formula.
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
        # Search space: free knobs are bm (rows/block = warps) and, for kernels
        # with a rollable ``:`` reduction, the reduction chunk plus the per-thread
        # vector width V. bn is pinned to 256 and bm capped at 16 (64*bm <= 1024)
        # by adjust_block_size_constraints. bm==1 rows may span W wavefronts
        # (thread_count = chunk // V, W = thread_count // 64); fp16/bf16 add V=8
        # (128-bit BufferCopy). No constexpr_range (that lands in PR4). FlyDSL has
        # no precompile and its JIT does not survive the subprocess benchmark
        # workers the generic search spawns, so enumerate the valid configs and
        # FiniteSearch them in-process.
        from ...runtime.config import Config

        spec = bound_kernel.config_spec
        default = spec.default_config()
        default_bs = default.config.get("block_sizes")
        if not isinstance(default_bs, list) or not default_bs:
            return default
        block_sizes = [int(b) for b in default_bs]
        ndim = len(block_sizes)

        row_hint = spec.block_sizes[0].size_hint if spec.block_sizes else 1
        candidates: list[Config] = []
        seen: set[tuple[object, ...]] = set()

        rl_ids = spec.reduction_loops.valid_block_ids()
        n_rl = len(rl_ids)

        # V=8 (128-bit BufferCopy) is valid only for 16-bit dtypes; fp32 tops out
        # at V=4 (a V=8 fp32 copy is 256-bit -> BackendUnsupported). Reduction
        # input dtype comes from the first tensor arg.
        _dtype = getattr(args[0], "dtype", None) if args else None
        _v_choices = (4, 8) if _dtype in (torch.float16, torch.bfloat16) else (4,)
        _hi_loop = self.max_reduction_loop() or 64

        user_tiled = _has_user_tiled_reduction(bound_kernel.env)
        # For user-tiled reductions, pin ALL column dims to 256 in the template
        # so every generated candidate (and the effective default) is valid.
        if user_tiled and ndim >= 2:
            for i in range(1, ndim):
                block_sizes[i] = 256

        # Numel of the first looped-reduction dim (used to filter OOB chunks).
        rl_numel: int | None = None
        if len(spec.reduction_loops):
            rl_numel = spec.reduction_loops[0].size_hint

        def _add(bs: list[int], rl: int | None, v: int | None = None) -> None:
            # Safety: for user-tiled reductions all column dims must be multiples
            # of 256 (one warp-pass = 64 lanes x 4 elems). Reject bad configs.
            if user_tiled and any(b % 256 != 0 for b in bs[1:]):
                return
            # Safety: reject a looped chunk whose last pass OOBs the divided
            # buffer (hardware buffer instructions fault on true OOB). The bound
            # is V-dependent (V = elems/thread), so it moves with the candidate's
            # V -- keep this in sync with the (chunk, V) enumeration below. When
            # ``v`` is None (bm>1 warp-per-row: one 64-lane warp folds the row),
            # memory_ops derives V = chunk // 64, so use that here too rather than
            # V=1 -- otherwise this models the undivided element space and the
            # bound diverges from the codegen it guards.
            if rl is not None and rl_numel is not None and rl_numel > 0:
                v_eff = v if v is not None else max(1, rl // 64)
                tc = rl // v_eff
                last_offset = ((rl_numel + rl - 1) // rl - 1) * rl
                max_div_idx = last_offset // v_eff + tc - 1
                n_div = (rl_numel + v_eff - 1) // v_eff
                if max_div_idx >= n_div:
                    return
            key = (tuple(bs), rl, v)
            if key in seen:
                return
            seen.add(key)
            kw: dict[str, Any] = {"block_sizes": list(bs)}
            if rl is not None:
                kw["reduction_loops"] = [rl] * n_rl
                if v is not None:
                    kw["cute_vector_widths"] = [v] * n_rl
            candidates.append(Config(**kw))

        for bm in (1, 2, 4, 8, 16):
            if bm > max(row_hint, 1):
                continue
            bs = list(block_sizes)
            bs[0] = bm
            if ndim >= 2:
                for i in range(1, ndim):
                    bs[i] = 256
            _add(bs, None)  # persistent
            if not rl_ids:
                continue
            if bm == 1:
                # bm==1: one row may span W wavefronts. thread_count = chunk // V,
                # W = thread_count // 64 (<=16). Enumerate (chunk, V): chunk from
                # 64*V (W=1) up to 1024*V (W=16), V in {4} (+8 for fp16/bf16).
                for v in _v_choices:
                    c = 64 * v
                    while c <= _hi_loop and (c // v) <= 1024:
                        _add(bs, c, v)
                        c *= 2
            else:
                # bm>1: one warp/row (thread_count 64), V = chunk // 64 derived
                # from the chunk. Offer chunk = 64*V for V in {1,2,4} (+8 fp16).
                _bm_vs = {1, 2, 4} | ({8} if 8 in _v_choices else set())
                for v in sorted(_bm_vs):
                    c = 64 * v
                    if c <= _hi_loop:
                        _add(bs, c, None)

        if not candidates:
            return default
        if len(candidates) == 1:
            return candidates[0]

        # Benchmark in-process: FlyDSL's JIT/HIP-stream state does not survive
        # the precompile/benchmark subprocess workers, so disable both paths.
        # These describe a permanent flydsl capability (its JIT can NEVER use the
        # subprocess workers), so they stay disabled for the bind -- not restored,
        # unlike baseline_fn below. In-place mutation mirrors the base autotune().
        bound_kernel.settings.autotune_precompile = None
        bound_kernel.settings.autotune_benchmark_subprocess = False

        # The default config (used as autotune baseline) may carry a
        # reduction_loops chunk that OOBs for non-power-of-2 N; use the
        # persistent (looped-off) config as the baseline so the accuracy check
        # compares against a safe reference. Unlike the two settings above this is
        # a temporary per-autotune override, so restore it in the finally so it
        # does not leak into a later explicit compile_config on the same bind.
        prev_baseline_fn = bound_kernel.settings.autotune_baseline_fn
        if prev_baseline_fn is None and rl_numel is not None:
            safe_bs = (
                [block_sizes[0]] + ([256] * (ndim - 1))
                if ndim >= 2
                else [block_sizes[0]]
            )
            persistent_config = Config(block_sizes=safe_bs)
            persistent_fn = bound_kernel.compile_config(
                persistent_config, allow_print=False
            )
            bound_kernel.settings.autotune_baseline_fn = lambda *a, _fn=persistent_fn: (
                _fn(*a)
            )

        from ...autotuner import FiniteSearch

        try:
            return FiniteSearch(bound_kernel, args, candidates).autotune()
        finally:
            bound_kernel.settings.autotune_baseline_fn = prev_baseline_fn

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from ...language import memory_ops

        # Reset per-compilation state so helpers are re-emitted on each compile.
        self._flydsl_needs_warp_helpers = False
        self._flydsl_helpers_emitted = False
        # Two regimes, encoded in block_sizes:
        #   W=1 (small-N): [bm, 256] -> bm rows/block, 1 warp/row, block = 64*bm.
        #   W>1 (large-N): [1, ...]  -> 1 row/block, W warps cooperate, 64*W,
        #                              W = thread_count // 64 from (chunk, V).
        bs = config.block_sizes or [1]
        bm = int(bs[0])

        # Guard: for user-tiled (explicit hl.tile(n)) reductions every column
        # block size must be a multiple of 256 (= V*64 = one warp-pass width).
        # The lane index formula ``offset//4 + lane`` goes OOB for any other bn.
        # This fires for neighbor-explored configs (e.g. bn=128 from autotune)
        # that the autotune() candidate filter cannot reject because they are
        # generated after the candidate list is built.
        from ..compile_environment import CompileEnvironment

        _env = CompileEnvironment.current()
        if _has_user_tiled_reduction(_env) and any(int(b) % 256 != 0 for b in bs[1:]):
            raise exc.BackendUnsupported(
                self.name,
                f"explicit hl.tile(n) reduction: column block size must be a "
                f"multiple of 256 (got block_sizes={list(bs)})",
            )

        # Whole-row looped reduction: thread_count = 64*W from (chunk, V). W>1
        # means W wavefronts cooperate on one row (bm==1) via the smem block
        # reduce; else one warp per row -> 64*bm.
        _tc = self._flydsl_looped_thread_count(config, bm)
        W = _tc // 64 if _tc is not None else 1
        self._flydsl_warps_per_row = W
        self._flydsl_num_threads = 64 * W if W > 1 else 64 * bm

        # Reset per-compile; every load/store tensor takes the vectorized buffer path.
        self._tensor_use_buffer = {}
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

                # Invariant: codegen looks this up by id(state.proxy_arg(0)),
                # which must be the SAME FakeTensor object as node.meta["val"]
                # here. If that ever breaks, the lookup misses and the load/store
                # silently takes the scalar path (wrong indexing, not an error).
                self._tensor_use_buffer[id(tensor)] = True

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        # Row (grid) tile -> warps: warp w = thread_idx.x // 64 owns its row.
        # Always dim x (flat block), regardless of the axis Helion assigns.
        if block_size_var == "1":
            return offset_var
        return f"({offset_var}) + fx.thread_idx.x // 64"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        # Column (loop) tile -> lanes. chunk = col_offset//vec + lane_id;
        # one warp (64 lanes) per row so lane_id = thread_idx.x % 64.
        # The literal 4 is the fp16 vec width; fp32 columns would need the width
        # derived from element bits (see _flydsl_buffer_setup). fp16-only for now.
        if block_size_var == "1":
            return offset_var
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
        # Column lane chunk: element_offset//vec + lane_id (one warp/row).
        # The literal 4 is the fp16 vec width (fp16-only; fp32 needs the derived
        # width, see _flydsl_buffer_setup).
        return f"{offsets_var} = ({lid}) // 4 + fx.thread_idx.x % 64"

    def range_str(self, begin: str | None, end: str, step: str | None) -> str | None:
        # Runtime scf.for. The step may be a module-level literal variable
        # (e.g. _REDUCTION_BLOCK_1) which scf.for materialises as an SSA value.
        # (PR4 adds the range_constexpr register-caching path.)
        #
        # Invariant: flydsl reaches range_str ONLY for the whole-row rolled
        # reduction loop (a LoopedReductionStrategy device loop); the outer grid
        # is driven off fx.block_idx, not a range(). We therefore emit a runtime
        # range() unconditionally, ignoring static_ranges/unroll config. This
        # method's string-only args cannot see the loop type to assert on, so the
        # invariant is enforced by convention today -- a range_str that keys off
        # the loop kind lands with the second flydsl loop path (PR4).
        args = [a for a in [begin, end] if a is not None]
        if step and step != "1":
            args.append(step)  # keep step as-is
        return "range(" + ", ".join(args) + ")"

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        # Lane mask (flat block, dim x): one warp (64 lanes) per row.
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
                return _flydsl_minimum_expr(a, b)

            @staticmethod
            def where(a: str, b: str, c: str) -> str:
                return f"({a}).select({b}, {c})"

            def __getattr__(self, name: str) -> object:
                # Any un-overridden op would fall through to Triton syntax
                # (tl.libdevice.*) and fail at trace time. Fail loudly instead.
                if name.startswith("_"):
                    raise AttributeError(name)
                raise exc.BackendUnsupported(backend_name, f"op {name!r}")

        return FlyDSLOpOverrides()

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        from ..ast_extension import statement_from_string

        # Emit the warp-reduce helper defs once, and only for kernels that
        # actually reduce (reduction_expr sets _needs_warp_helpers). This hook
        # runs per scalar arg after the kernel body is generated, so the flag is
        # already set by the time we get here; a pure-elementwise kernel leaves
        # it False and emits nothing.
        if self._flydsl_helpers_emitted or not self._flydsl_needs_warp_helpers:
            return []
        self._flydsl_helpers_emitted = True

        # Intra-warp fold: one warp per row -> warp shuffle covers 64 lanes, no
        # smem. The 6 XOR-shuffle folds (log2 64) are unrolled here rather than
        # emitted as a range_constexpr loop (that register-caching path lands in
        # PR4). The warp helpers are needed by BOTH regimes: W=1 uses them
        # directly; the W>1 block reduce calls them for the intra-warp step.
        def _fold(step_op: str) -> str:
            return "\n".join(
                f"    w = {step_op.format(off=64 >> (s + 1))}" for s in range(6)
            )

        _wmax_body = _fold("w.maximumf(w.shuffle_xor({off}, 64))")
        _wsum_body = _fold(
            "w.addf(w.shuffle_xor({off}, 64), fastmath=arith.FastMathFlags.fast)"
        )
        stmts: list[ast.AST] = [
            statement_from_string(h)
            for h in [
                f"def _flydsl_wmax(w):\n{_wmax_body}\n    return w",
                f"def _flydsl_wmin(w):\n    w = -w\n{_wmax_body}\n    return -w",
                f"def _flydsl_wsum(w):\n{_wsum_body}\n    return w",
            ]
        ]

        if self._flydsl_warps_per_row <= 1:
            return stmts

        # W>1: W warps cooperate on one row -> cross-warp block reduce via smem.
        # Each warp folds its 64 lanes with the warp helper, lane 0 writes the
        # partial to slot ``_wave`` of a fp32 smem buffer, then warp 0 folds the
        # W partials and broadcasts the result via the disjoint result slot.
        _WARP = 64
        _SLOTS = max(1, self._flydsl_num_threads // _WARP)  # one partial per wave
        _RESULT = _SLOTS  # result slot, disjoint from partials [0.._SLOTS-1]
        _TOTAL = _SLOTS + 1

        stmts.extend(
            [
                statement_from_string(
                    f"""@fx.struct
class _FlyDSLRedBuf:
    s: fx.Array[fx.Float32, {_TOTAL}, 16]"""
                ),
                statement_from_string(
                    "_flydsl_lds = fx.SharedAllocator().allocate(_FlyDSLRedBuf).peek()"
                ),
                statement_from_string(
                    f"_flydsl_sred = _flydsl_lds.s.view(fx.make_layout({_TOTAL}, 1))"
                ),
            ]
        )

        # ``_ident`` seeds the inactive lanes of warp 0's final fold with the
        # reduction identity so out-of-range slots don't perturb the result.
        _block = [
            ("_flydsl_bmax", "_flydsl_wmax", "fx.Float32(-3.4028235e+38)"),
            ("_flydsl_bmin", "_flydsl_wmin", "fx.Float32(3.4028235e+38)"),
            ("_flydsl_bsum", "_flydsl_wsum", "fx.Float32(0.0)"),
        ]
        for _bname, _wname, _ident in _block:
            stmts.append(
                statement_from_string(
                    f"""def {_bname}(w, _sred):
    gpu.barrier()
    _lane = fx.thread_idx.x % {_WARP}
    _wave = fx.thread_idx.x // {_WARP}
    _nwaves = fx.block_dim.x // {_WARP}
    _r = {_wname}(w)
    if _lane == 0:
        fx.memref_store(_r, _sred, _wave)
    gpu.barrier()
    if _wave == 0:
        _in = _lane < _nwaves
        _ls = _in.select(_lane, 0)
        _v = fx.memref_load(_sred, _ls)
        _vv = _in.select(_v, {_ident})
        _vv = {_wname}(_vv)
        if _lane == 0:
            fx.memref_store(_vv, _sred, {_RESULT})
    gpu.barrier()
    _out = fx.memref_load(_sred, {_RESULT})
    gpu.barrier()
    return _out"""
                )
            )
        return stmts

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        # Looped-reduction accumulate step. ``acc`` is a per-thread vector
        # (wrap_reduction_accumulator seeds it as fx.Vector.filled before the
        # loop); the freshly-loaded ``val`` goes on the LEFT so the final
        # _flydsl_wsum/_wmax fold calls .reduce() on a vector.
        #
        # Both operands must share the accumulator dtype (e.g. fp16 input folded
        # into an fp32 acc), so cast the freshly-loaded value first.
        val = self.cast_expr(val, self.dtype_str(dtype))
        # ``maximumf``/``minimumf`` require both operands to be the SAME vector
        # shape. Re-broadcast acc to ``val``'s shape with ``Vector.filled_like(
        # val, 0) + acc`` -- a shape-only zero vector plus acc -- NOT
        # ``(val) - (val)``: a masked tail reduction feeds ``val = +/-inf``
        # (identity from _mask_to), and ``inf - inf`` is NaN. This is a no-op when
        # acc already matches val's shape. ``sum`` uses ``+`` directly (it
        # promotes either way; the masked tail feeds a finite 0 identity).
        if reduction_type == "sum":
            return f"({val}) + ({acc})"
        acc_bc = f"(fx.Vector.filled_like({val}, 0) + ({acc}))"
        if reduction_type == "max":
            return f"({val}).maximumf({acc_bc})"
        if reduction_type == "min":
            # flydsl Vector has no ``.minimumf``; use min(a,b) = -max(-a,-b).
            return f"(-((-({val})).maximumf((-({acc_bc})))))"
        raise exc.BackendUnsupported(self.name, f"reduction combine {reduction_type!r}")

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
        # The _flydsl_{w,b}{sum,max,min} helper defs these calls reference are
        # emitted by scalar_arg_preamble (which runs after this body is
        # generated); flag that they are needed so they are emitted only for
        # reducing kernels.
        if reduction_type not in ("sum", "max", "min"):
            raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")
        self._flydsl_needs_warp_helpers = True
        # fast-math is applied to ``sum`` only, on purpose: float add is not
        # associative and the shuffle fold reorders it, so ``fast`` authorizes
        # that reordering; max/min are order-independent and need no flag.
        if self._flydsl_warps_per_row == 1:
            # W=1: one warp per row -> warp shuffle covers the whole row, no smem.
            if reduction_type == "sum":
                return f"_flydsl_wsum({input_name}.reduce(ReductionOp.ADD, fastmath=arith.FastMathFlags.fast))"
            if reduction_type == "max":
                return f"_flydsl_wmax({input_name}.reduce(ReductionOp.MAX))"
            return f"_flydsl_wmin({input_name}.reduce(ReductionOp.MIN))"
        # W>1: W warps cooperate on one row -> cross-warp block reduce via shared
        # mem. The cross-warp buffer ``_flydsl_sred`` is fx.Float32, so the
        # per-thread ``.reduce(...)`` scalar MUST be fp32 before it is stored to /
        # reloaded from smem. An fp16 reduce result (e.g. ``torch.amax`` on an
        # fp16 tile) round-trips through the fp32 smem incorrectly and corrupts
        # the fold -> systematically wrong max/min. ``.to(fx.Float32)`` fixes it;
        # it is a no-op for the already-fp32 sum accumulator.
        if reduction_type == "sum":
            return f"_flydsl_bsum(({input_name}.reduce(ReductionOp.ADD, fastmath=arith.FastMathFlags.fast)).to(fx.Float32), _flydsl_sred)"
        if reduction_type == "max":
            return f"_flydsl_bmax(({input_name}.reduce(ReductionOp.MAX)).to(fx.Float32), _flydsl_sred)"
        return f"_flydsl_bmin(({input_name}.reduce(ReductionOp.MIN)).to(fx.Float32), _flydsl_sred)"

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
        return _flydsl_minimum_expr(a, b)
