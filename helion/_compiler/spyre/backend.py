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

#: A Spyre launch costs milliseconds and the device is a shared, serialized
#: resource, so keep the sample count low: autotuning sweeps dozens of configs.
_BENCH_WARMUP = 1
_BENCH_REPS = 3


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
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.emitter = SpyreEmitter()
        #: id(fake tensor) -> stick axis of the real input it stands for.
        self._input_stick_dims: dict[int, int] = {}

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

    def supports_precompile(self) -> bool:
        return False

    def get_interleaved_bench(self) -> object:
        """Wall-clock interleaved timing; the default needs Triton.

        ``autotuner/benchmarking.py:383`` imports ``triton.runtime`` for its
        cache-clearing and device-interface handles, which do not exist here.
        The hook's own docstring (`_compiler/backend.py:603-604`) says backends
        without Triton event timing should override.

        Interleaving exists to cancel slow drift between candidates, so the
        sweeps are round-robin over the candidate functions rather than
        candidate-at-a-time.
        """

        def spyre_interleaved_bench(
            fns: object,
            *,
            repeat: int = 3,
            desc: object = None,
            max_total_ms: object = None,
            **kwargs: object,
        ) -> list[float]:
            import time

            fns = list(fns)
            for fn in fns:
                fn()

            totals = [0.0] * len(fns)
            sweeps = max(1, min(int(repeat), _BENCH_REPS))
            for _ in range(sweeps):
                for i, fn in enumerate(fns):
                    start = time.perf_counter()
                    fn()
                    totals[i] += (time.perf_counter() - start) * 1e3
            return [t / sweeps for t in totals]

        return spyre_interleaved_bench

    def get_do_bench(self) -> object:
        """Wall-clock timing: the default ``do_bench`` is CUDA-shaped.

        ``autotuner/benchmarking.py`` times with CUDA events and streams
        (`:105-114`), which do not exist for the ``spyre`` device. Spyre
        launches are synchronous from the host's point of view -- ``runner.run``
        returns once the jobplan completes -- so a monotonic clock around the
        call is the honest measurement here, with no device-side timer to read.
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
            import time

            for _ in range(_BENCH_WARMUP):
                fn()

            samples: list[float] = []
            for _ in range(_BENCH_REPS):
                start = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - start) * 1e3)

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
        except Exception:
            # A dtype or layout the backend cannot describe fails later, at the
            # point where it is actually used and can be named.
            pass

    def input_stick_dim(self, fake: torch.Tensor) -> int | None:
        return self._input_stick_dims.get(id(fake))

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
