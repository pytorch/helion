"""Interface for out-of-tree Helion codegen backends (experimental).

Out-of-tree backends import Helion internals only from this module and, for
now, pin one Helion minor version.  ``BACKEND_API_VERSION`` covers a defined
subset: every name here stays importable, plus the ``Backend`` hooks and
strategy methods in ``COVERED_BACKEND_HOOKS`` and ``COVERED_STRATEGY_METHODS``
(described in ``docs/api/backend_api.md``).  What
``InductorOpOverrides`` aliases (torch's private ``OpsHandler[Any]``) is not
covered.  Other attributes of the classes re-exported here are not covered and
may change.

Backends attach ``@codegen`` handlers to public ops through
``helion.language`` (e.g. ``hl.load``); ``reduce_op`` and ``mask_to_op`` are
the internal ops a backend may also need.
"""

from __future__ import annotations

from typing import Any as _Any

from torch._inductor.ops_handler import OpsHandler as _OpsHandler

from . import exc as exc
from ._compiler.ast_extension import expr_from_string as expr_from_string
from ._compiler.ast_extension import statement_from_string as statement_from_string
from ._compiler.aten_lowering import Lowering as Lowering
from ._compiler.backend import Backend as Backend
from ._compiler.backend import BackendState as BackendState
from ._compiler.backend_registry import BACKEND_API_VERSION as BACKEND_API_VERSION
from ._compiler.backend_registry import COVERED_BACKEND_HOOKS as COVERED_BACKEND_HOOKS
from ._compiler.backend_registry import (
    COVERED_STRATEGY_METHODS as COVERED_STRATEGY_METHODS,
)
from ._compiler.backend_registry import (
    register_compiler_backend as register_compiler_backend,
)
from ._compiler.compile_environment import CompileEnvironment as CompileEnvironment
from ._compiler.compile_environment import _symint_sympy_expr as symint_sympy_expr
from ._compiler.device_function import Argument as Argument
from ._compiler.device_function import DeviceBodyContext as DeviceBodyContext
from ._compiler.device_function import DeviceFunction as DeviceFunction
from ._compiler.device_ir import DeviceIR as DeviceIR
from ._compiler.device_ir import ForLoopGraphInfo as ForLoopGraphInfo
from ._compiler.device_ir import GraphInfo as GraphInfo
from ._compiler.device_ir import IfGraphInfo as IfGraphInfo
from ._compiler.device_ir import ReductionLoopGraphInfo as ReductionLoopGraphInfo
from ._compiler.host_function import HostFunction as HostFunction
from ._compiler.inductor_lowering import CodegenState as CodegenState
from ._compiler.reduction_strategy import (
    LoopedReductionStrategy as LoopedReductionStrategy,
)
from ._compiler.reduction_strategy import (
    PersistentReductionStrategy as PersistentReductionStrategy,
)
from ._compiler.reduction_strategy import ReductionStrategy as ReductionStrategy
from ._compiler.tile_dispatch import TileStrategyDispatch as TileStrategyDispatch
from ._compiler.tile_strategy import CompactedShape as CompactedShape
from ._compiler.tile_strategy import DeviceGridState as DeviceGridState
from ._compiler.tile_strategy import DeviceLoopState as DeviceLoopState
from ._compiler.tile_strategy import TileStrategy as TileStrategy
from ._compiler.triton.backend import TritonBackend as TritonBackend
from .autotuner import FiniteSearch as FiniteSearch
from .autotuner.config_fragment import ConfigSpecFragment as ConfigSpecFragment
from .autotuner.config_spec import BlockSizeSpec as BlockSizeSpec
from .autotuner.config_spec import ConfigSpec as ConfigSpec
from .autotuner.config_spec import VectorWidthSpec as VectorWidthSpec
from .language._decorators import codegen as codegen
from .language._tracing_ops import _mask_to as mask_to_op
from .language.reduce_ops import _reduce as reduce_op
from .runtime.config import Config as Config
from .runtime.kernel import BoundKernel as BoundKernel

# torch's private OpsHandler; only the name is covered (Backend.inductor_op_overrides).
InductorOpOverrides = _OpsHandler[_Any]

__all__ = [
    "BACKEND_API_VERSION",
    "COVERED_BACKEND_HOOKS",
    "COVERED_STRATEGY_METHODS",
    "Argument",
    "Backend",
    "BackendState",
    "BlockSizeSpec",
    "BoundKernel",
    "CodegenState",
    "CompactedShape",
    "CompileEnvironment",
    "Config",
    "ConfigSpec",
    "ConfigSpecFragment",
    "DeviceBodyContext",
    "DeviceFunction",
    "DeviceGridState",
    "DeviceIR",
    "DeviceLoopState",
    "FiniteSearch",
    "ForLoopGraphInfo",
    "GraphInfo",
    "HostFunction",
    "IfGraphInfo",
    "InductorOpOverrides",
    "LoopedReductionStrategy",
    "Lowering",
    "PersistentReductionStrategy",
    "ReductionLoopGraphInfo",
    "ReductionStrategy",
    "TileStrategy",
    "TileStrategyDispatch",
    "TritonBackend",
    "VectorWidthSpec",
    "codegen",
    "exc",
    "expr_from_string",
    "mask_to_op",
    "reduce_op",
    "register_compiler_backend",
    "statement_from_string",
    "symint_sympy_expr",
]
