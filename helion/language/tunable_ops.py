from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import AutoSize
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.type_propagation import TileIndexType
from .._compiler.type_propagation import TypeInfo
from .._compiler.type_propagation import _to_proxy
from ..autotuner.config_fragment import BaseIntegerFragment
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_fragment import assert_integer_power_of_two
from ..autotuner.config_spec import VALID_KEYS
from ..exc import NotInsideKernel
from . import _decorators
from .loops import _normalize_begin_end

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = [
    "register_block_size",
    "register_tunable",
]


@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_block_size(min_or_max: int, max_or_none: int | None = None, /) -> int:
    """
    Explicitly register a block size that should be autotuned and can be used for
    allocations and inside hl.tile(..., block_size=...).

    This is useful if you have two loops where you want them to share a block size,
    or if you need to allocate a kernel tensor before the hl.tile() loop.

    The signature can one of:
        hl.register_block_size(max)
        hl.register_block_size(min, max)

    Where min and max are integers that control the range of block_sizes searched by
    the autotuner.  Max may be a symbolic shape, but min must be a constant integer.
    """
    raise exc.NotInsideKernel


@_decorators.ref(register_block_size)
def _(min_or_max: int, max_or_none: int | None = None, /) -> int:
    # In ref mode, always return the maximum value (full dimension size)
    if max_or_none is None:
        return min_or_max
    return max_or_none


@_decorators.type_propagation(register_block_size)
def _(
    min_or_max: TypeInfo, max_or_none: TypeInfo | None = None, /, *, origin: Origin
) -> TypeInfo:
    from .._compiler.type_propagation import BlockSizeType

    min_type, max_type = _normalize_begin_end(min_or_max, max_or_none, origin=origin)
    min_proxy = _to_proxy(min_type)
    max_proxy = _to_proxy(max_type)
    if not isinstance(max_proxy, (int, torch.SymInt)):
        raise exc.IncorrectTileUsage(
            f"expected max to be an integer or size, got {max_proxy!s}"
        )
    if not isinstance(min_proxy, int):
        raise exc.IncorrectTileUsage(
            f"expected min to be an integer constant, got {min_proxy!s}"
        )
    env = CompileEnvironment.current()
    result = TileIndexType.allocate(AutoSize(), origin)
    loop_spec = env.config_spec.block_sizes.block_id_lookup(result.block_id)
    loop_spec.min_size = assert_integer_power_of_two(max(1, min_proxy))
    loop_spec.max_size = next_power_of_2(env.size_hint(max_proxy))
    block_id = result.block_id
    return BlockSizeType(origin, env.block_sizes[block_id].var, block_id)


def _block_id_from_state(state: CodegenState) -> int:
    """Extract the block_id from the current state for nodes hl.register_block_size."""
    from .._compiler.type_propagation import BlockSizeType
    from .._compiler.type_propagation import SymIntType

    env = CompileEnvironment.current()
    if state.fx_node is not None:
        val = state.fx_node.meta["val"]
        if isinstance(val, BlockSizeType):
            return val.block_id
        assert isinstance(val, SymIntType)
        block_id = env.get_block_id(val.value)
        assert block_id is not None
        return block_id
    current_node = ExtendedAST.current()[-1]
    type_info = current_node._type_info
    if isinstance(type_info, BlockSizeType):
        return type_info.block_id
    assert isinstance(type_info, SymIntType)
    block_id = env.get_block_id(type_info.value)
    assert block_id is not None
    return block_id


@_decorators.codegen(register_block_size)
def _(state: CodegenState) -> ast.AST:
    env = CompileEnvironment.current()
    block_size = env.config_spec.block_sizes.config_get(
        state.config.block_sizes, _block_id_from_state(state)
    )
    assert block_size is not None
    return expr_from_string(constant_repr(block_size))


@_decorators.api(is_device_only=False)
def register_tunable(name: str, fragment: ConfigSpecFragment) -> int:
    """
    Register a tunable parameter for autotuning.

    This function allows you to define parameters that can be automatically tuned
    during the autotuning process. The fragment defines the search space and default value.

    Args:
        name: The key for the tunable parameter in the Config().
        fragment: A ConfigSpecFragment that defines the search space (e.g., PowerOfTwoFragment)

    Returns:
        int: The value assigned to this tunable parameter in the current configuration.
    """
    raise NotInsideKernel


@_decorators.type_propagation(register_tunable)
def _register_tunable_type(
    name: TypeInfo, fragment: TypeInfo, *, origin: Origin
) -> TypeInfo:
    # During type propagation, register the tunable parameter and return unbacked symint
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import NumericType

    env = CompileEnvironment.current()

    try:
        fragment_val = fragment.as_literal()
        name_val = name.as_literal()
    except NotImplementedError:
        fragment_val = None
        name_val = None
    if not (isinstance(name_val, str) and isinstance(fragment_val, ConfigSpecFragment)):
        raise exc.RegisterTunableArgTypes(name, fragment)
    del name, fragment

    if name_val in VALID_KEYS or f"{name_val}s" in VALID_KEYS:
        raise exc.TunableNameConflict(name_val)
    if (
        name_val in env.config_spec.user_defined_tunables
        and env.config_spec.user_defined_tunables[name_val] != fragment_val
    ):
        raise exc.TunableNameConflict(name_val)

    # register the value for tuning
    env.config_spec.user_defined_tunables[name_val] = fragment_val

    python_type = type(fragment_val.default())
    if not issubclass(python_type, (int, float, bool)):
        raise exc.TunableTypeNotSupported(python_type)
    return NumericType.subtype(python_type).new_unbacked(origin)


@_decorators.codegen(register_tunable)
def _register_tunable_codegen(state: CodegenState) -> ast.AST:
    name = state.proxy_arg(0)
    assert isinstance(name, str)
    config_value = state.config[name]
    assert isinstance(config_value, (int, float, bool))
    return expr_from_string(constant_repr(config_value))


@_decorators.ref(register_tunable)
def _(name: str, fragment: ConfigSpecFragment) -> int:
    """Reference implementation of register_tunable."""
    from ..runtime.ref_mode import RefModeContext

    # Get config from the current RefModeContext
    context = RefModeContext.current()
    config = context.config

    # Determine the value to use
    value: object
    assert config is not None
    if name in config:
        value = config[name]
    else:
        value = fragment.default()

    # For BaseIntegerFragment subclasses (IntegerFragment, PowerOfTwoFragment), apply clamp
    # This ensures the value is within valid bounds
    if isinstance(fragment, BaseIntegerFragment) and isinstance(value, (int, bool)):
        value = fragment.clamp(int(value))

    # Convert to int if needed
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    # For other types (like float), convert to int
    return int(value)  # type: ignore[arg-type]
