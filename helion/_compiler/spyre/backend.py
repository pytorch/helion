"""``SpyreBackend``: a Helion backend emitting torch-spyre OpSpecs.

Plan 1.1: the emission seam is the *per-node lowering* one
(inductor_lowering.py:1642-1662), where a codegen result that is not an
``ast.AST`` passes straight through to the interpreter env. So each compute node
returns an ``OpSpec``-producing ``Operand`` rather than an expression string, and
``inductor_op_overrides`` is deliberately left unimplemented -- a node that
reaches Inductor's own pointwise fallback is a bug and must fail loudly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import torch

from ... import exc
from ..backend import Backend
from .emitter import SpyreEmitter

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import Lowering
    from ..tile_dispatch import TileStrategyDispatch


#: Widest stick any supported dtype uses: 64 elements at fp16/bf16
#: (torch_spyre._C.get_elem_in_stick), 32 at fp32. Used as the floor for a
#: trailing block size so no config can split a stick.
_MAX_ELEMS_PER_STICK = 64

#: Hard ceiling on concurrent LX intermediates. A **count**, not a byte budget:
#: 3 concurrent LX buffers fail at launch at both 8 KB each (64x64 fp16) and
#: 64 KB each (128x256 fp16) -- an 8x footprint difference with the same wall.
#: The nominal budget (_LX_TRACKER_CAPACITY_BYTES * dxp_lx_frac_avail, ~406 KB,
#: scratchpad/allocator.py:169,:1524) would allow ~6 and is not a safe guide.
_MAX_LX_BUFFERS = 2

#: A Spyre launch costs milliseconds and the device is a shared, serialized
#: resource, so keep the sample count low: autotuning sweeps dozens of configs.
_BENCH_WARMUP = 1
_BENCH_REPS = 3

#: Launches submitted per timed sample, before one synchronize drains them.
#:
#: A Spyre launch is **asynchronous**: ``spyre_stream.cpp:261`` wraps
#: ``launchOperationCompute``, so timing a call measures the enqueue and the
#: loop overlaps submit(N+1) with compute(N). Timing one call at a time
#: therefore measures the *queue*, not the kernel -- it reports ~26-30 us for
#: matmuls from 0.07 to 4.3 GFLOP, a 64x work range, which at the top end would
#: be 158 TFLOP/s. It also **inverts** the ranking the autotuner needs: on a
#: 512x512 matmul it rates the 1-core config at 18 us and the 16-core config at
#: 23-60 us, when draining shows 181 us against 62 us.
#:
#: Batching before the synchronize amortizes the one-off sync cost while still
#: emptying the queue, so a per-launch figure stays meaningful. 8 is enough to
#: separate every config of that matmul with disjoint IQRs while keeping a
#: sweep affordable: it costs 8 launches per sample, not 8x the wall time,
#: because the submissions pipeline.
_BENCH_BATCH = 8


def _drained(fn: object, batch: int = _BENCH_BATCH) -> float:
    """Microseconds for one launch of ``fn``, with the device queue drained.

    Submits ``batch`` launches, synchronizes once, and divides. The
    synchronize is what makes this a measurement of the device rather than of
    the submission path.
    """
    import time

    import torch

    start = time.perf_counter()
    for _ in range(batch):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e3 / batch


def _sencores() -> int:
    """Cores available to one kernel (``SENCORES``, default 32)."""
    from torch_spyre._inductor import config as spyre_config

    return int(spyre_config.sencores)


#: Device for a synthesized scalar constant in the generated host wrapper.
_DEVICE_EXPR = '"spyre"' 


class SpyreBackend(Backend):
    """Emits torch-spyre ``OpSpec``/``LoopSpec`` objects per Device IR node."""

    _SUPPORTED_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "block_sizes",
            "range_unroll_factors",
            "spyre_lx_buffers",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.emitter = SpyreEmitter()
        #: id(fake tensor) -> stick axis of the real input it stands for.
        self._input_stick_dims: dict[int, int] = {}
        #: id(fake tensor) -> host rank, so output_stick_dim can match by rank.
        self._input_ranks: dict[int, int] = {}

    @property
    def name(self) -> str:
        return "spyre"

    @property
    def experimental(self) -> bool:
        return True

    def validate_environment(self) -> None:
        try:
            import torch_spyre  # noqa: F401
            from torch_spyre._inductor.op_spec import OpSpec  # noqa: F401
        except ImportError as e:
            raise exc.BackendUnsupported(
                self.name,
                "torch-spyre is not installed; install it with: pip install torch-spyre",
            ) from e

    # -- what Helion must not assume on this backend ----------------------

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        # Plan 4.3: Spyre wants stick multiples, not powers of two.
        return False

    @property
    def requires_shape_specialized_module(self) -> bool:
        # Plan 5.2: the generated module caches a compiled Spyre kernel keyed by
        # spec hash, so distinct static_shapes specializations that happen to
        # emit identical source must not share it.
        return True

    def supports_config_key(self, key: str) -> bool:
        return key in self._SUPPORTED_CONFIG_KEYS

    def backend_tunables(self) -> dict[str, object]:
        """Expose LX residency to the autotuner instead of hard-coding it.

        How many intermediates live in LX before the rest spill to ``hbm_pool``
        is a real performance decision, and the right value is kernel-dependent
        -- so it must be searched, not baked in. Nothing in the stack diagnoses
        asking for too much: the specs validate, the bundle builds, the backend
        compiler succeeds, and the kernel then fails *at launch* with
        ``RAS::RUNTIMESCHEDULER::ComputeHardwareError``, which also poisons the
        device stream for the rest of the process.

        The ceiling is a **count, not a byte budget**: measured on this device,
        3 concurrent LX intermediates fail at both 8 KB each (64x64 fp16) and
        64 KB each (128x256 fp16), an 8x difference in footprint with the same
        wall. So the search space is capped at ``_MAX_LX_BUFFERS`` and
        ``autotune_config_is_viable`` rejects anything above it, rather than
        letting the autotuner discover the limit as a hardware fault.
        """
        from ...autotuner.config_spec import IntegerFragment

        return {
            # low=0 (everything in hbm_pool) .. high=_MAX_LX_BUFFERS, default 2.
            "spyre_lx_buffers": IntegerFragment(
                0, _MAX_LX_BUFFERS, _MAX_LX_BUFFERS
            ),
        }

    def supports_precompile(self) -> bool:
        return False

    def get_interleaved_bench(self) -> object:
        """Interleaved timing with the device queue drained; the default needs
        Triton.

        ``autotuner/benchmarking.py:383`` imports ``triton.runtime`` for its
        cache-clearing and device-interface handles, which do not exist here.
        The hook's own docstring (`_compiler/backend.py:603-604`) says backends
        without Triton event timing should override.

        Interleaving exists to cancel slow drift between candidates, so the
        sweeps are round-robin over the candidate functions rather than
        candidate-at-a-time. Each candidate's turn is a *drained* batch (see
        ``_BENCH_BATCH``): without the synchronize this timed the submission
        queue and ranked candidates backwards.
        """

        def spyre_interleaved_bench(
            fns: object,
            *,
            repeat: int = 3,
            desc: object = None,
            max_total_ms: object = None,
            **kwargs: object,
        ) -> list[float]:
            fns = list(fns)
            for fn in fns:
                fn()

            totals = [0.0] * len(fns)
            sweeps = max(1, min(int(repeat), _BENCH_REPS))
            for _ in range(sweeps):
                for i, fn in enumerate(fns):
                    totals[i] += _drained(fn)
            return [t / sweeps for t in totals]

        return spyre_interleaved_bench

    def get_do_bench(self) -> object:
        """Wall-clock timing with the device queue drained.

        ``autotuner/benchmarking.py`` times with CUDA events and streams
        (`:105-114`), which do not exist for the ``spyre`` device, and there is
        no device-side timer to read instead.

        A Spyre launch is asynchronous, so the naive form of this -- a clock
        around one call -- measures submission and is flat in the problem size.
        Each sample is a drained batch instead; see ``_BENCH_BATCH`` for the
        measurements that forced it.
        """

        def spyre_do_bench(
            fn: object,
            warmup: int = 25,
            rep: int = 100,
            grad_to_none: object = None,
            quantiles: object = None,
            return_mode: str = "mean",
            process_group_name: object = None,
            **kwargs: object,
        ) -> float | tuple[float, ...]:
            import statistics

            for _ in range(_BENCH_WARMUP):
                _drained(fn)

            samples: list[float] = [_drained(fn) for _ in range(_BENCH_REPS)]

            if return_mode == "all":
                return tuple(samples)
            if quantiles:
                ordered = sorted(samples)
                return tuple(
                    ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]
                    for q in quantiles
                )
            reducer = {
                "min": min,
                "max": max,
                "median": statistics.median,
                "mean": statistics.fmean,
            }.get(return_mode, statistics.fmean)
            return float(reducer(samples))

        return spyre_do_bench

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        """Keep the autotuner off block sizes that split a stick.

        Plan 4.3: the trailing axis is the within-stick axis and is never
        divided -- "a stick is the unit of transfer, so splitting it across
        cores would split a stick" (opspec_utils.py:515-516). A sub-stick
        trailing block is not caught by ``validate_op_specs``; it surfaces as an
        opaque ``SubprocException`` from the backend compiler, so bound the
        search instead of discovering it late.

        The floor is the widest stick any supported dtype uses (64 elements at
        fp16/bf16; fp32 uses 32 and is therefore also satisfied).

        The same floors also keep the **core budget** reachable. The product of
        the per-dim splits is the core count, bounded by ``sencores`` (32), and
        ``autotune_config_is_viable`` already rejects candidates that overshoot
        -- but the autotuner's *reference* config is taken straight from these
        fragment defaults (``benchmark_provider.py:876``) and never goes through
        that check. On 512x512 the default landed on ``block_sizes=[32, 64]``,
        which asks for 16*8 = 128 cores and dies in ``dbo-opt`` with SIGABRT.
        The generic recovery cannot help: it *shrinks* block sizes
        (``_compute_reference_baseline``), and on a fixed extent a smaller block
        means **more** cores, so every retry moves further over budget (128 ->
        256 -> 512). Raising the floors here fixes it at the source, since
        ``BlockSizeSpec._fragment`` clamps its default to ``min_size``.
        """
        from ...autotuner.config_spec import BlockSizeSpec

        specs = [s for s in block_specs if isinstance(s, BlockSizeSpec)]
        if len(specs) < 2:
            return
        for trailing in specs[1:]:
            # Never raise the floor above the dim itself: a dim narrower than
            # one stick (e.g. N=32 fp16) occupies a single partial stick, which
            # is not a split and is therefore legal. Forcing 64 there makes the
            # block indivisible by the extent and nothing can satisfy it.
            extent = getattr(trailing, "size_hint", None)
            if isinstance(extent, int) and extent < _MAX_ELEMS_PER_STICK:
                continue
            trailing.update_min(_MAX_ELEMS_PER_STICK)

        self._raise_floors_for_core_budget(specs)

    @staticmethod
    def _raise_floors_for_core_budget(specs: list[object]) -> None:
        """Raise block-size floors until the minimum core count fits sencores.

        At each floor the split of a dim is ``extent // min_size``, so the
        product over dims is the fewest cores any in-bounds config can ask for.
        While that exceeds the budget, double the floor of the dim currently
        contributing the most splits -- halving its core count -- which
        converges and keeps the floors as low as the budget allows.
        """
        budget = _sencores()

        def splits(spec: object) -> int:
            extent = int(getattr(spec, "size_hint", 0) or 0)
            floor = max(1, int(getattr(spec, "min_size", 1) or 1))
            return max(1, extent // floor) if extent > 0 else 1

        for _ in range(32):
            counts = [splits(s) for s in specs]
            cores = 1
            for c in counts:
                cores *= c
            if cores <= budget:
                return
            # Double the floor of the worst offender; a power-of-two floor is
            # required (update_min asserts it).
            worst = specs[counts.index(max(counts))]
            floor = max(1, int(getattr(worst, "min_size", 1) or 1))
            extent = int(getattr(worst, "size_hint", 0) or 0)
            if floor >= extent:
                # Already one core on this dim; nothing further to give.
                return
            worst.update_min(floor * 2)

    def autotune_config_is_viable(
        self, config_spec: object, config: Config
    ) -> bool:
        """Reject configs Spyre cannot express, before anything is emitted.

        Three rules, each one a hard constraint rather than a preference:

        * **Exact division.** Spyre work division and coarse-tile counts are
          integers; a block size that does not divide its extent has no valid
          ``iteration_space`` split (`wsr/coarse_tile.py:1384-1395` raises on
          this for coarse tiles). The emitter already fails loudly, but a
          rejected *config* must never reach codegen.
        * **Stick alignment.** As in ``adjust_block_size_constraints``: a
          trailing block below one stick splits a stick.
        * **One reduction split.** Plan 4.1: at most one reduction dim may be
          split across cores (`work_division.py:317-322`), since a K-split
          builds a PSUM chain the emitter does not model.
        * **Core budget.** The product of the per-dim splits is the number of
          cores the kernel asks for, bounded by ``config.sencores``
          (`_inductor/config.py:143`, default 32; also the ``max_cores`` the
          work divider itself uses at `work_division.py:1990`). Overshooting is
          not caught by ``validate_op_specs`` -- it surfaces as an opaque
          ``SubprocException`` from the backend compiler, which an autotune
          sweep hits constantly as it shrinks block sizes.
        """
        lx_buffers = config.config.get("spyre_lx_buffers")
        if lx_buffers is not None and not (0 <= int(lx_buffers) <= _MAX_LX_BUFFERS):
            # Above the measured ceiling the kernel fails at launch, not here.
            return False

        block_sizes = config.block_sizes
        if not block_sizes:
            return True

        # config_spec is the authority here, not CompileEnvironment.current():
        # this hook is called from the autotuner's search loop, where there is
        # no active compile environment, so reading the ambient one silently
        # accepted every candidate.
        spec = getattr(config_spec, "block_sizes", None)
        if spec is None:
            return True
        infos = [spec.block_id_lookup(bid) for bid in spec.valid_block_ids()]
        reduction_splits = 0
        cores = 1
        for position, (info, block) in enumerate(zip(infos, block_sizes)):
            extent = int(getattr(info, "size_hint", 0) or 0)
            value = int(block)
            if extent <= 0:
                continue
            if value <= 0 or extent % value != 0:
                # Work division must be exact.
                return False
            if (
                position > 0
                and value < _MAX_ELEMS_PER_STICK
                and extent > _MAX_ELEMS_PER_STICK
            ):
                # A trailing block below one stick splits a stick -- unless the
                # whole dim is narrower than a stick, where it cannot.
                return False
            if getattr(info, "reduction", False) and extent // value > 1:
                reduction_splits += 1
            cores *= extent // value

        # At most one reduction dim may be split across cores.
        if reduction_splits > 1:
            return False
        return cores <= _sencores()

    # -- the emission seam ------------------------------------------------

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        self.emitter.reset()

    def record_input_layout(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> None:
        """Remember each real input's stick axis before it is fakified.

        Plan 5.3 says input layouts are *read*, not chosen -- but by codegen
        time every tensor is a ``FakeTensor`` whose ``device_tensor_layout()``
        returns ``None``, so the real layout is unrecoverable there. Without
        this the backend silently assumed the default trailing-dim stick axis
        for every input, which is wrong numerics (not a crash) for a
        transposed operand and means no ReStickifyOpHBM is ever emitted.
        """
        from .layout import stick_dim_of

        try:
            self._input_stick_dims[id(fake)] = stick_dim_of(real)
            self._input_ranks[id(fake)] = real.dim()
        except Exception:
            # A dtype or layout the backend cannot describe fails later, at the
            # point where it is actually used and can be named.
            pass

    def input_stick_dim(self, fake: torch.Tensor) -> int | None:
        return self._input_stick_dims.get(id(fake))

    def output_stick_dim(self, rank: int) -> int | None:
        """Stick axis every rank-``rank`` kernel tensor agrees on, if any.

        A kernel's output tensor is recorded by ``record_input_layout`` like any
        other argument (``torch.empty_like(x)`` in the kernel body is still a
        host tensor passed to the launcher), but which recorded entry *is* the
        output is not known here. When every rank-matching tensor shares one
        stick axis, that axis is unambiguous and is the right target; a mixture
        means a genuine conflict, which the relayout path resolves per operand.
        """
        axes = {
            dim
            for key, dim in self._input_stick_dims.items()
            if self._input_ranks.get(key) == rank
        }
        return axes.pop() if len(axes) == 1 else None

    def pre_inductor_lowering(self, node: torch.fx.Node) -> Lowering | None:
        """Install a Spyre lowering for every compute node.

        Preempts Inductor's ``PointwiseLowering``/``ReductionLowering``, which
        would funnel the node's result through ``_unpack_opsvalue``
        (inductor_lowering.py:1415-1419) and assert on a non-string.
        """
        if node.op != "call_function":
            return None
        from .compute_ops import SpyreOpLowering
        from .optable import spyre_op_name

        if spyre_op_name(node.target) is None:
            return None
        return SpyreOpLowering(self, node.target)

    # -- module shape (plan 5.1) -----------------------------------------

    # -- the vestigial device body ---------------------------------------
    #
    # Plan 5.1: for Spyre there is no device Python -- the payload is the spec
    # literal. Helion still emits a device FunctionDef and walks the tile
    # strategy to build it, so these hooks exist only to keep that walk
    # producing *syntactically legal* Python that is never executed. Nothing
    # here feeds an OpSpec: the iteration space is derived from the node's own
    # shape in compute_ops, not from these index variables.

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        return "int"

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"_spyre_pid({dim})"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"({offset_var})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"({offset_var})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"({offsets_var})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return expr_str

    def expands_broadcast_dims(self) -> bool:
        # Shape lives in device_size/device_coordinates, not in an expression,
        # so a Triton-style [None, :] broadcast-expand is meaningless here.
        # Same move FlyDSL makes (flydsl/backend.py:294).
        return False

    @property
    def function_decorator(self) -> str:
        return ""

    def function_decorator_for_args(self, args: Sequence[object]) -> str:
        """Emit the spec literal, then decorate nothing.

        Plan 5.1: this runs from ``codegen_function_def`` -- after the device
        body has been walked (so the spec tree is complete) and while
        ``module_statements`` is still open (so the literal can be inserted
        without a core edit). The literal itself is printed by torch-spyre's own
        ``_codegen_op_spec_list`` (spyre_kernel.py:1572); plan 5.1 is explicit
        that we must not write a second printer.
        """
        from ..ast_extension import statement_from_string
        from ..device_function import DeviceFunction
        from .spec_source import spec_source

        device_function = DeviceFunction.current()
        emitter = self.emitter
        specs = emitter.specs()
        if not specs:
            raise exc.BackendUnsupported(
                self.name, "kernel emitted no OpSpecs"
            )

        name = device_function.name
        statements = device_function.codegen.module_statements
        statements.append(
            statement_from_string(f"_SPYRE_SPECS = {spec_source(specs)}")
        )
        statements.append(
            statement_from_string(f"_SPYRE_POOL_SIZE = {emitter.pool_size}")
        )
        del name
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_spyre_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "torch": "import torch",
            "sympify": "from sympy import sympify",
            "DataFormats": "from torch_spyre._C import DataFormats",
            "ElementArrangement": "from torch_spyre._C import ElementArrangement",
            "OpSpec": "from torch_spyre._inductor.op_spec import OpSpec",
            "LoopSpec": "from torch_spyre._inductor.op_spec import LoopSpec",
            "TensorArg": "from torch_spyre._inductor.op_spec import TensorArg",
            "IndirectAccess": (
                "from torch_spyre._inductor.op_spec import IndirectAccess"
            ),
            "_default_spyre_launcher": (
                "from helion.runtime import default_spyre_launcher"
                " as _default_spyre_launcher"
            ),
        }

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        return frozenset({"_spyre_specs", "_spyre_pool_size"})

    # -- hooks that are structurally wrong here --------------------------
    #
    # Each of these is a *string*-valued expression hook. On this backend a
    # value is a buffer, not an expression, so a string here would be silently
    # meaningless. Plan 6.4: the carry is structural, emitted by the
    # _for_loop handler, so the reduction string hooks stay unsupported.

    def dtype_str(self, dtype: torch.dtype) -> str:
        from .layout import device_dtype

        return str(device_dtype(dtype))

    def acc_type(self, dtype: torch.dtype) -> str:
        raise exc.BackendUnsupported(
            self.name, "acc_type: Spyre accumulators are buffers, not type strings"
        )

    def inductor_op_overrides(self) -> Any:
        # Plan 1.1: deliberately unimplemented. Reaching this means a compute
        # node escaped pre_inductor_lowering.
        raise exc.BackendUnsupported(
            self.name,
            "inductor_op_overrides: every compute node must be claimed by"
            " pre_inductor_lowering",
        )

    def reduction_combine_expr(self, *args: object, **kwargs: object) -> str:
        raise exc.BackendUnsupported(
            self.name, "reduction_combine_expr: the carry is an explicit buffer"
        )

    def reduction_expr(self, *args: object, **kwargs: object) -> str:
        raise exc.BackendUnsupported(
            self.name, "reduction_expr: the carry is an explicit buffer"
        )

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: Sequence[object] | None = None,
    ) -> list[str]:
        """Pass the host tensors positionally, in OpSpec ``arg_index`` order.

        The device body is vestigial here, so it references none of the kernel
        tensors and Helion registers no device args for them: ``args``,
        ``tensor_host_args`` and ``sorted_args`` all arrive empty, and the
        emitted call would be ``_launcher(kernel, grid)`` with no tensors --
        which fails at runtime with "symbolic_args tensor_id out of range".

        The emitter's own buffer table is the source of truth instead. Each
        graph input/output was registered from Helion's ``tensor_arg`` table, so
        its ``name`` *is* the host wrapper's variable name and its ``arg_index``
        is the position SDSC expects. A scalar constant the emitter synthesized
        (an accumulator zero-fill has no host tensor) is materialized here.
        """
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")

        consts = {index for index, _, _ in self.emitter.const_args}
        launch: list[tuple[int, str]] = []
        for operand in self.emitter.args:
            if operand.arg_index < 0:
                continue
            if operand.arg_index in consts:
                launch.append(
                    (
                        operand.arg_index,
                        f"torch.zeros((), dtype={operand.dtype!r},"
                        f" device={_DEVICE_EXPR})",
                    )
                )
            elif operand.name:
                launch.append((operand.arg_index, operand.name))
            else:
                raise exc.BackendUnsupported(
                    self.name, f"kernel arg {operand.arg_index} has no host name"
                )
        return [name for _, name in sorted(launch)]

    def scalar_arg_preamble(self, arg: object) -> list[ast.AST]:
        return []
