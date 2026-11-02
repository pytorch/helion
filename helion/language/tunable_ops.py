from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import AutoSize
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import ConfigValueExpression
from .._compiler.compile_environment import _symint_expr
from .._compiler.type_info import TileIndexType
from .._compiler.type_info import TypeInfo
from .._compiler.type_info import _to_proxy
from ..autotuner.config_fragment import BaseIntegerFragment
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_fragment import assert_integer_power_of_two
from ..autotuner.config_spec import GRF_MODE_KEY
from ..autotuner.config_spec import VALID_KEYS
from ..exc import NotInsideKernel
from . import _decorators
from .loops import _normalize_begin_end

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = [
    "persistent_num_workers",
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
    the autotuner.  Both may be symbolic shapes, in which case their size hints are
    used to bound the search.
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
    from .._compiler.type_info import BlockSizeType

    min_type, max_type = _normalize_begin_end(min_or_max, max_or_none, origin=origin)
    min_proxy = _to_proxy(min_type)
    max_proxy = _to_proxy(max_type)
    if not isinstance(max_proxy, (int, torch.SymInt)):
        raise exc.IncorrectTileUsage(
            f"expected max to be an integer or size, got {max_proxy!s}"
        )
    if not isinstance(min_proxy, (int, torch.SymInt)):
        raise exc.IncorrectTileUsage(
            f"expected min to be an integer or size, got {min_proxy!s}"
        )
    env = CompileEnvironment.current()
    # Host scalars (globals, int arguments, `*args` unpacked from a tuple) are
    # faked into SymInts during type propagation, so min may arrive symbolic even
    # when the user wrote a constant.  Both bounds only shape the autotuner search
    # space, so resolve them to hints the same way.
    min_proxy = env.size_hint(min_proxy)
    existing_block_id = (
        env.get_block_id(max_proxy) if isinstance(max_proxy, torch.SymInt) else None
    )
    if existing_block_id is None:
        result = TileIndexType.allocate(AutoSize(), origin)
        block_id = result.block_id
    else:
        block_id = existing_block_id
    loop_spec = env.config_spec.block_sizes.block_id_lookup(block_id)
    loop_spec.min_size = assert_integer_power_of_two(max(1, min_proxy))
    loop_spec.max_size = next_power_of_2(env.size_hint(max_proxy))
    loop_spec.declared_max_size = loop_spec.max_size
    return BlockSizeType(origin, env.block_sizes[block_id].var, block_id)


def _block_id_from_state(state: CodegenState) -> int:
    """Extract the block_id from the current state for nodes hl.register_block_size."""
    from .._compiler.type_info import BlockSizeType
    from .._compiler.type_info import SymIntType

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


@_decorators.codegen(register_block_size, "common")
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
    from .._compiler.type_info import NumericType

    env = CompileEnvironment.current()

    try:
        fragment_val = fragment.as_literal()
        name_val = name.as_literal()
    except NotImplementedError:
        fragment_val = None
        name_val = None
    if not (isinstance(name_val, str) and isinstance(fragment_val, ConfigSpecFragment)):
        raise exc.RegisterTunableArgTypes(name, fragment)

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
    result = NumericType.subtype(python_type).new_unbacked(origin)
    # Codegen uses this provenance to annotate device arguments as constexpr.
    for symbol in result.value._sympy_().free_symbols:
        assert isinstance(symbol, sympy.Symbol)
        env.tunable_symbols.add(symbol)
    if isinstance(result.value, torch.SymInt):
        expr = _symint_expr(result.value)
        if expr is not None:
            env.config_value_expressions[expr] = ConfigValueExpression(
                "config", (name_val,)
            )
    return result


@_decorators.codegen(register_tunable, "common")
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


def computes_persistent_num_workers(device: torch.device) -> bool:
    """Whether :func:`persistent_num_workers` scales the SM count on ``device``.

    On XPU, get_num_sm() is the Xe-core count, but one work-group per Xe-core
    often leaves hardware threads idle: how many fit depends on num_warps and
    the GRF mode.  Elsewhere one program per SM is kept.
    """
    return device.type == "xpu"


@_decorators.api(is_device_only=False)
def persistent_num_workers(device: torch.device, *, reserved_sms: int = 0) -> int:
    """
    Number of workers for a persistent kernel that loops over ``hl.grid(n)``.

    On XPU, several work-groups can share an Xe-core, so this is
    ``helion.runtime.get_num_sm(device, reserved_sms=reserved_sms)`` times the
    number of work-groups that fit on one Xe-core, as computed by
    :func:`helion.runtime.get_xpu_programs_per_xecore` from ``num_warps`` and
    the ``grf_mode`` config value.  The kernel is compiled with that GRF mode,
    which is autotuned over :func:`helion.runtime.get_xpu_grf_modes`.
    Elsewhere it is ``get_num_sm()``.

    Args:
        device: Device the kernel runs on.
        reserved_sms: SMs to leave free for other work, as in
            :func:`helion.runtime.get_num_sm`.

    Returns:
        int: The worker count for the current configuration.
    """
    raise NotInsideKernel


@_decorators.type_propagation(persistent_num_workers)
def _persistent_num_workers_type(
    device: TypeInfo, *, reserved_sms: TypeInfo | None = None, origin: Origin
) -> TypeInfo:
    from .._compiler.type_info import SymIntType
    from ..autotuner.config_fragment import EnumFragment
    from ..runtime import get_num_sm
    from ..runtime.triton.launcher import get_xpu_eu_resources
    from ..runtime.triton.launcher import get_xpu_grf_modes

    env = CompileEnvironment.current()
    device_val = device.as_literal()
    reserved_val = 0 if reserved_sms is None else reserved_sms.as_literal()
    assert isinstance(device_val, torch.device)
    assert isinstance(reserved_val, int)
    num_sm = get_num_sm(device_val, reserved_sms=reserved_val)
    if not computes_persistent_num_workers(device_val):
        return TypeInfo.from_example(num_sm, origin)

    # The worker count needs the GRF size before compiling, so leave out 0
    # (Triton's automatic choice) from the grf_mode backend tunable.
    env.config_spec.backend_tunable_fragments[GRF_MODE_KEY] = EnumFragment(
        get_xpu_grf_modes(device_val)
    )
    result = SymIntType(origin, env.create_unbacked_symint(hint=num_sm))
    expr = _symint_expr(result.value)
    assert isinstance(expr, sympy.Symbol)
    env.tunable_symbols.add(expr)
    # Mirrors get_xpu_programs_per_xecore().
    eus, max_threads, grf_per_eu = get_xpu_eu_resources(device_val)
    threads_per_eu = ConfigValueExpression(
        "floordiv", (grf_per_eu, ConfigValueExpression("config", (GRF_MODE_KEY,)))
    )
    threads = ConfigValueExpression(
        "mul", (eus, ConfigValueExpression("min", (max_threads, threads_per_eu)))
    )
    programs = ConfigValueExpression(
        "max",
        (
            ConfigValueExpression(
                "floordiv", (threads, ConfigValueExpression("config", ("num_warps",)))
            ),
            1,
        ),
    )
    env.config_value_expressions[expr] = ConfigValueExpression(
        "mul", (num_sm, programs)
    )
    return result


def _check_explicit_grf_mode(grf_mode: object) -> None:
    if not grf_mode:
        raise exc.InvalidConfig(
            "hl.persistent_num_workers() computes the worker count from the GRF "
            f"size, so it needs a non-zero grf_mode, got {grf_mode!r}"
        )


@_decorators.codegen(persistent_num_workers, "common")
def _persistent_num_workers_codegen(state: CodegenState) -> ast.AST:
    # Defaulted arguments are plain constants; only user-written ones need visiting.
    device, reserved_sms = [
        state.codegen.visit(arg) if isinstance(arg, ExtendedAST) else arg
        for arg in (state.ast_arg(0), state.ast_arg(1))
    ]
    call = expr_from_string(
        "helion.runtime.get_num_sm({device}, reserved_sms={reserved_sms})",
        device=device,
        reserved_sms=reserved_sms,
    )
    if not computes_persistent_num_workers(CompileEnvironment.current().device):
        return call
    grf_mode = state.config.get(GRF_MODE_KEY)
    _check_explicit_grf_mode(grf_mode)
    return expr_from_string(
        "{call} * helion.runtime.get_xpu_programs_per_xecore("
        f"{{device}}, num_warps={state.config.num_warps}, grf_mode={grf_mode})",
        call=call,
        device=device,
    )


@_decorators.ref(persistent_num_workers)
def _(device: torch.device, reserved_sms: int = 0) -> int:
    from ..runtime import Config
    from ..runtime import get_num_sm
    from ..runtime import get_xpu_programs_per_xecore
    from ..runtime.ref_mode import RefModeContext
    from ..runtime.triton.launcher import get_xpu_grf_modes

    num_sm = get_num_sm(device, reserved_sms=reserved_sms)
    if not computes_persistent_num_workers(device):
        return num_sm
    config = RefModeContext.current().config or Config()
    grf_mode = config.get(GRF_MODE_KEY, get_xpu_grf_modes(device)[0])
    _check_explicit_grf_mode(grf_mode)
    assert isinstance(grf_mode, int)
    return num_sm * get_xpu_programs_per_xecore(
        device, num_warps=config.num_warps, grf_mode=grf_mode
    )
