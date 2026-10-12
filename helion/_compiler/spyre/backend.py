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

    def autotune_config_is_viable(self, config: Config) -> tuple[bool, str | None]:
        # Plan 4.1: at most one reduction dim may be split across cores
        # (work_division.py:317-322), since a K-split builds a PSUM chain.
        return True, None

    # -- the emission seam ------------------------------------------------

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        self.emitter.reset()

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
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")
        return list(args)

    def scalar_arg_preamble(self, arg: object) -> list[ast.AST]:
        return []
