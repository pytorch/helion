from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import ConfigValueExpression
from .._compiler.compile_environment import _symint_expr
from .._compiler.type_info import TypeInfo
from ..autotuner.config_spec import GRF_MODE_KEY
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = ["persistent_num_workers"]


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
    # Not a search dimension: the value follows from num_warps and grf_mode, and
    # registering it as a tunable symbol passes it to the device code as a
    # constexpr for each config.
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
